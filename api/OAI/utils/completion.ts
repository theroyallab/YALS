import { SSEStreamingApi } from "hono/streaming";
import { Model } from "@/bindings/bindings.ts";
import { GenerationChunk } from "@/bindings/types.ts";
import { HonoRequest } from "hono";
import {
    createUsageStats,
    GenerationType,
    staticGenerate,
} from "@/api/OAI/utils/generation.ts";

import {
    CompletionRequest,
    CompletionRespChoice,
    CompletionResponse,
} from "../types/completions.ts";
import { toGeneratorError } from "@/common/networking.ts";
import { CancellationError } from "@/common/errors.ts";
import { logger } from "@/common/logging.ts";

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
    requestId: string,
    stream: SSEStreamingApi,
    model: Model,
    params: CompletionRequest,
) {
    logger.info(`Received streaming completion request ${requestId}`);

    const abortController = new AbortController();
    let finished = false;

    // If an abort happens before streaming starts
    req.raw.signal.addEventListener("abort", () => {
        if (!finished) {
            abortController.abort(
                new CancellationError(
                    `Streaming completion ${requestId} cancelled by user.`,
                ),
            );
            finished = true;
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

            await stream.writeSSE({
                data: JSON.stringify(streamChunk),
            });
        }

        logger.info(`Finished streaming completion request ${requestId}`);
    } catch (error) {
        await stream.writeSSE({
            data: JSON.stringify(toGeneratorError(error)),
        });
    }

    finished = true;
}

export async function generateCompletion(
    req: HonoRequest,
    requestId: string,
    model: Model,
    params: CompletionRequest,
) {
    logger.info(`Received completion request ${requestId}`);

    // Handle generation in the common function
    const gen = await staticGenerate(
        req,
        requestId,
        GenerationType.Completion,
        model,
        params.prompt,
        params,
    );

    const response = createResponse(gen, model.path.name);

    return response;
}
