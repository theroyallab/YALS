import { Queue } from "@core/asyncutil";
import { SSEStreamingApi } from "hono/streaming";

import {
    convertFinishReason,
    createUsageStats,
    GenerationType,
    staticGenerate,
    streamCollector,
} from "@/api/OAI/utils/generation.ts";
import { GenerationChunk } from "@/bindings/types.ts";
import { CancellationError } from "@/common/errors.ts";
import { toGeneratorError } from "@/common/networking.ts";
import { logger } from "@/common/logging.ts";
import {
    CompletionRequest,
    CompletionRespChoice,
    CompletionResponse,
} from "../types/completions.ts";
import { OAIContext } from "../types/context.ts";

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
    ctx: OAIContext,
    params: CompletionRequest,
    stream: SSEStreamingApi,
) {
    logger.info(`Received streaming completion request ${ctx.requestId}`);

    const genAbortController = new AbortController();
    let finished = false;

    // If an abort happens before streaming starts
    ctx.cancellationSignal.addEventListener("abort", () => {
        if (!finished) {
            genAbortController.abort(
                new CancellationError(
                    `Streaming completion ${ctx.requestId} cancelled by user.`,
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
                ctx,
                params.prompt,
                params,
                genAbortController.signal,
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

            const chunk = await queue.pop({
                signal: genAbortController.signal,
            });
            if (chunk instanceof Error) {
                genAbortController.abort();
                throw chunk;
            }

            const streamChunk = createResponse([chunk], ctx.model.path.name);
            await stream.writeSSE({ data: JSON.stringify(streamChunk) });

            if (chunk.kind === "finish") {
                completedTasks++;
            }

            if (completedTasks === params.n && queue.size === 0) {
                logger.info(
                    `Finished streaming completion request ${ctx.requestId}`,
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
    ctx: OAIContext,
    params: CompletionRequest,
) {
    logger.info(`Received completion request ${ctx.requestId}`);

    // Handle generation in the common function
    const generations = await staticGenerate(
        ctx,
        GenerationType.Completion,
        params.prompt,
        params,
    );

    const response = createResponse(generations, ctx.model.path.name);

    logger.info(`Finished completion request ${ctx.requestId}`);
    return response;
}
