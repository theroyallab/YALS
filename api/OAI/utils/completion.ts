import { SSEStreamingApi } from "hono/streaming";
import { Model } from "@/bindings/bindings.ts";
import { GenerationChunk } from "@/bindings/types.ts";
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

function createResponse(chunk: GenerationChunk, modelName: string) {
    const choice = CompletionRespChoice.parse({
        text: chunk.text,
    });

    const usage = chunk.kind === "finish" ? createUsageStats(chunk) : undefined;

    const response = CompletionResponse.parse({
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
            const streamChunk = createResponse(chunk, model.path.name);

            stream.onAbort(() => {
                if (!finished) {
                    abortController.abort();
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
    const response = createResponse(gen, model.path.name);

    return response;
}
