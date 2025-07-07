import { UsageStats } from "@/api/OAI/types/completions.ts";
import { Model } from "@/bindings/bindings.ts";
import { FinishChunk } from "@/bindings/types.ts";
import { logger } from "@/common/logging.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";
import { toHttpException } from "@/common/networking.ts";
import { CancellationError } from "@/common/errors.ts";

export enum GenerationType {
    Completion = "Completion",
    ChatCompletion = "Chat completion",
}

export function createUsageStats(chunk: FinishChunk) {
    const usage = UsageStats.parse({
        prompt_tokens: chunk.promptTokens,
        prompt_time: chunk.promptSec,
        prompt_tokens_per_sec: chunk.promptTokensPerSec,
        completion_tokens: chunk.genTokens,
        completion_time: chunk.genSec,
        completion_tokens_per_sec: chunk.genTokensPerSec,
        total_tokens: chunk.promptTokens + chunk.genTokens,
        total_time: chunk.totalSec,
    });

    return usage;
}

export function convertFinishReason(chunk: FinishChunk) {
    if (chunk.toolCalls) {
        return "tool_calls";
    } else if (chunk.finishReason === "MaxNewTokens") {
        return "length";
    } else {
        return "stop";
    }
}

export async function staticGenerate(
    requestId: string,
    genType: GenerationType,
    prompt: string,
    params: BaseSamplerRequest,
    model: Model,
    requestSignal: AbortSignal,
) {
    const abortController = new AbortController();
    let finished = false;

    requestSignal.addEventListener("abort", () => {
        if (!finished) {
            abortController.abort(
                new CancellationError(
                    `${genType} ${requestId} cancelled by user.`,
                ),
            );
            finished = true;
        }
    });

    try {
        const result = await model.generate(
            requestId,
            prompt,
            params,
            abortController.signal,
        );

        logger.info(`Finished ${genType.toLowerCase()} request ${requestId}`);

        finished = true;
        return result;
    } catch (error) {
        throw toHttpException(error);
    }
}
