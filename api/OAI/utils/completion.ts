import { SSEStreamingApi } from "hono/streaming";
import { Model } from "@/bindings/bindings.ts";

import {
    CompletionRequest,
    CompletionRespChoice,
    CompletionResponse,
} from "../types/completions.ts";
import { HonoRequest } from "hono";
import { logger } from "@/common/logging.ts";

async function createResponse(text: string, modelName: string) {
    const choice = await CompletionRespChoice.parseAsync({
        text: text,
    });

    const response = await CompletionResponse.parseAsync({
        choices: [choice],
        model: modelName,
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

    const generator = model.generateGen(
        params.prompt,
        params,
        abortController.signal,
    );

    for await (const chunk of generator) {
        stream.onAbort(() => {
            if (!finished) {
                abortController.abort();
                logger.error("Streaming completion aborted");
                finished = true;

                // Break out of the stream loop
                return;
            }
        });

        const streamChunk = await createResponse(chunk, model.path.name);

        await stream.writeSSE({
            data: JSON.stringify(streamChunk),
        });
    }
}

export async function generateCompletion(
    model: Model,
    params: CompletionRequest,
    req: HonoRequest,
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
        params.prompt,
        params,
        abortController.signal,
    );
    const response = await createResponse(result, model.path.name);
    finished = true;

    return response;
}
