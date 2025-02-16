import { HonoRequest } from "hono";

import { UsageStats } from "@/api/OAI/types/completions.ts";
import { Model } from "@/bindings/bindings.ts";
import { FinishChunk } from "@/bindings/types.ts";
import { logger } from "@/common/logging.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";
import { toHttpException } from "@/common/networking.ts";

export function createUsageStats(chunk: FinishChunk) {
    const usage = UsageStats.parse({
        prompt_tokens: chunk.promptTokens,
        completion_tokens: chunk.genTokens,
        total_tokens: chunk.promptTokens + chunk.genTokens,
    });

    return usage;
}

export async function staticGenerate(
    req: HonoRequest,
    model: Model,
    prompt: string,
    params: BaseSamplerRequest,
) {
    const abortController = new AbortController();
    let finished = false;

    req.raw.signal.addEventListener("abort", () => {
        if (!finished) {
            abortController.abort();
            logger.error("Completion aborted");
        }
    });

    try {
        const result = await model.generate(
            prompt,
            params,
            abortController.signal,
        );

        finished = true;
        return result;
    } catch (error) {
        throw toHttpException(error);
    }
}
