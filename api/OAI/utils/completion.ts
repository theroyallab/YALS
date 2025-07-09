import { SSEStreamingApi } from "hono/streaming";

import {
    convertFinishReason,
    createUsageStats,
    GenerationType,
    staticGenerate,
} from "@/api/OAI/utils/generation.ts";
import { Model } from "@/bindings/bindings.ts";
import { GenerationChunk } from "@/bindings/types.ts";
import { CancellationError } from "@/common/errors.ts";
import { toGeneratorError } from "@/common/networking.ts";
import { logger } from "@/common/logging.ts";
import {
    CompletionRequest,
    CompletionRespChoice,
    CompletionResponse,
} from "../types/completions.ts";

function createResponse(chunk: GenerationChunk, modelName: string) {
    const finishReason = chunk.kind === "finish"
        ? convertFinishReason(chunk)
        : undefined;
    const choice = CompletionRespChoice.parse({
        text: chunk.text,
        finish_reason: finishReason,
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
    requestId: string,
    stream: SSEStreamingApi,
    params: CompletionRequest,
    model: Model,
    requestSignal: AbortSignal,
) {
    logger.info(`Received streaming completion request ${requestId}`);

    const abortController = new AbortController();
    let finished = false;

    // If an abort happens before streaming starts
    requestSignal.addEventListener("abort", () => {
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
            requestId,
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
        await stream.writeSSE({ data: "[DONE]" });
    } catch (error) {
        await stream.writeSSE({
            data: JSON.stringify(toGeneratorError(error)),
        });
    }

    finished = true;
}

export async function generateCompletion(
    requestId: string,
    params: CompletionRequest,
    model: Model,
    requestSignal: AbortSignal,
) {
    logger.info(`Received completion request ${requestId}`);

    // Handle generation in the common function
    const gen = await staticGenerate(
        requestId,
        GenerationType.Completion,
        params.prompt,
        params,
        model,
        requestSignal,
    );

    const response = createResponse(gen, model.path.name);

    return response;
}
