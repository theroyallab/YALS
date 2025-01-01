import { HonoRequest } from "hono";
import { Model } from "@/bindings/bindings.ts";
import { logger } from "@/common/logging.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";

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
