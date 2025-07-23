import { Queue } from "@core/asyncutil";
import { SSEStreamingApi } from "hono/streaming";

import {
    convertFinishReason,
    createUsageStats,
    GenerationType,
    staticGenerate,
    streamCollector,
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

function createResponse(chunks: GenerationChunk[], modelName: string) {
    const choices: CompletionRespChoice[] = [];
    for (const chunk of chunks) {
        const finishReason = chunk.kind === "finish"
            ? convertFinishReason(chunk)
            : undefined;

        const choice = CompletionRespChoice.parse({
            index: chunk.taskIdx,
            text: chunk.text,
            finish_reason: finishReason,
        });

        choices.push(choice);
    }

    const finalChunk = chunks.at(-1);
    const usage = finalChunk?.kind === "finish"
        ? createUsageStats(finalChunk)
        : undefined;

    const response = CompletionResponse.parse({
        choices,
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
        const queue = new Queue<GenerationChunk | Error>();
        const genTasks = [];

        for (let i = 0; i < params.n; i++) {
            const task = streamCollector(
                requestId,
                params.prompt,
                params,
                model,
                abortController.signal,
                i,
                queue,
            );

            genTasks.push(task);
        }

        let completedTasks = 0;
        while (true) {
            // Abort if the signal is set
            if (finished) {
                break;
            }

            const chunk = await queue.pop({ signal: abortController.signal });
            if (chunk instanceof Error) {
                abortController.abort();
                throw chunk;
            }

            const streamChunk = createResponse([chunk], model.path.name);
            await stream.writeSSE({ data: JSON.stringify(streamChunk) });

            if (chunk.kind === "finish") {
                completedTasks++;
            }

            if (completedTasks === params.n && queue.size === 0) {
                logger.info(
                    `Finished streaming completion request ${requestId}`,
                );
                await stream.writeSSE({ data: "[DONE]" });

                break;
            }
        }
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
    const generations = await staticGenerate(
        requestId,
        GenerationType.Completion,
        params.prompt,
        params,
        model,
        requestSignal,
    );

    const response = createResponse(generations, model.path.name);

    logger.info(`Finished completion request ${requestId}`);
    return response;
}
