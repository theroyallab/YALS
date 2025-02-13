import { SSEStreamingApi } from "hono/streaming";
import { GenerationChunk, Model } from "@/bindings/bindings.ts";
import { HonoRequest } from "hono";
import {
    createUsageStats,
    staticGenerate,
} from "@/api/OAI/utils/generation.ts";
import { logger } from "@/common/logging.ts";

import {
    CompletionRequest,
    CompletionRespChoice,
    CompletionResponse,
} from "../types/completions.ts";
import { toGeneratorError } from "@/common/networking.ts";

async function createResponse(chunk: GenerationChunk, modelName: string) {
    const choice = await CompletionRespChoice.parseAsync({
        text: chunk.text,
    });

    const usage = chunk.kind === "finish"
        ? await createUsageStats(chunk)
        : undefined;

    const response = await CompletionResponse.parseAsync({
        choices: [choice],
        model: modelName,
        usage,
    });

    return response;
}

export async function streamCompletion(
    req: HonoRequest,
    stream: SSEStreamingApi,
    model: Model,
    params: CompletionRequest,
) {
    const abortController = new AbortController();
    let finished = false;

    // If an abort happens before streaming starts
    req.raw.signal.addEventListener("abort", () => {
        if (!finished) {
            abortController.abort();
            logger.error("Streaming completion aborted");
        }
    });

    try {
        const generator = model.generateGen(
            params.prompt,
            params,
            abortController.signal,
        );

        for await (const chunk of generator) {
            const streamChunk = await createResponse(chunk, model.path.name);

            stream.onAbort(() => {
                if (!finished) {
                    abortController.abort();
                    logger.error("Streaming completion aborted");
                    finished = true;

                    // Break out of the stream loop
                    return;
                }
            });

            await stream.writeSSE({
                data: JSON.stringify(streamChunk),
            });
        }
    } catch (error) {
        await stream.writeSSE({
            data: JSON.stringify(toGeneratorError(error)),
        });
    }

    finished = true;
}

export async function generateCompletion(
    req: HonoRequest,
    model: Model,
    params: CompletionRequest,
) {
    const gen = await staticGenerate(req, model, params.prompt, params);
    const response = await createResponse(gen, model.path.name);

    return response;
}
