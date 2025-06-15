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
        completion_tokens: chunk.genTokens,
        total_tokens: chunk.promptTokens + chunk.genTokens,
    });

    return usage;
}

export function convertFinishReason(chunk: FinishChunk) {
    return chunk.finishReason === "MaxNewTokens" ? "length" : "stop";
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
