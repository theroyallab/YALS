import { HonoRequest } from "hono";
import { FinishChunk, Model } from "@/bindings/bindings.ts";
import { logger } from "@/common/logging.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";
import { UsageStats } from "@/api/OAI/types/completions.ts";

export async function createUsageStats(chunk: FinishChunk) {
    const usage = await UsageStats.parseAsync({
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

    const result = await model.generate(
        prompt,
        params,
        abortController.signal,
    );

    finished = true;
    return result;
}
