import {
    CommonCompletionRequest,
    UsageStats,
} from "@/api/OAI/types/completions.ts";
import { FinishChunk, GenerationChunk } from "@/bindings/types.ts";
import { toHttpException } from "@/common/networking.ts";
import { CancellationError } from "@/common/errors.ts";
import { Queue } from "@core/asyncutil/queue";
import { OAIContext } from "../types/context.ts";

export enum GenerationType {
    Completion = "Completion",
    ChatCompletion = "Chat completion",
}

export function createUsageStats(chunk: FinishChunk) {
    const usage = UsageStats.parse({
        prompt_tokens: chunk.promptTokens,
        prompt_time: chunk.promptSec,
        prompt_tokens_per_sec: chunk.promptTokensPerSec,
        completion_tokens: chunk.genTokens,
        completion_time: chunk.genSec,
        completion_tokens_per_sec: chunk.genTokensPerSec,
        total_tokens: chunk.promptTokens + chunk.genTokens,
        total_time: chunk.totalSec,
    });

    return usage;
}

export function convertFinishReason(chunk: FinishChunk) {
    if (chunk.toolCalls) {
        return "tool_calls";
    } else if (chunk.finishReason === "MaxNewTokens") {
        return "length";
    } else {
        return "stop";
    }
}

export async function staticGenerate(
    ctx: OAIContext,
    genType: GenerationType,
    prompt: string,
    params: CommonCompletionRequest,
) {
    const abortController = new AbortController();
    let finished = false;

    ctx.cancellationSignal.addEventListener("abort", () => {
        if (!finished) {
            abortController.abort(
                new CancellationError(
                    `${genType} ${ctx.requestId} cancelled by user.`,
                ),
            );
            finished = true;
        }
    });

    try {
        const genTasks: Promise<FinishChunk>[] = [];

        for (let i = 0; i < params.n; i++) {
            const genRequestId = params.n > 1
                ? `${ctx.requestId}-${i}`
                : ctx.requestId;

            const task = ctx.model.generate(
                genRequestId,
                prompt,
                params,
                abortController.signal,
                i,
            );

            genTasks.push(task);
        }

        const genResults = await Promise.allSettled(genTasks);
        const generations = genResults.reduce((acc, result) => {
            if (result.status === "rejected") {
                throw result.reason instanceof Error
                    ? result.reason
                    : new Error(result.reason);
            }

            acc.push(result.value);
            return acc;
        }, [] as FinishChunk[]);

        finished = true;
        return generations;
    } catch (error) {
        throw toHttpException(error);
    }
}

export async function streamCollector(
    ctx: OAIContext,
    prompt: string,
    params: CommonCompletionRequest,
    genAbortSignal: AbortSignal,
    taskIdx: number,
    genQueue: Queue<GenerationChunk | Error>,
) {
    try {
        const genRequestId = params.n > 1
            ? `${ctx.requestId}-${taskIdx}`
            : ctx.requestId;

        const generator = ctx.model.generateGen(
            genRequestId,
            prompt,
            params,
            genAbortSignal,
            taskIdx,
        );

        for await (const chunk of generator) {
            genQueue.push(chunk);

            // Break on finish
            if (chunk.kind === "finish") {
                break;
            }
        }
    } catch (ex) {
        genQueue.push(ex as Error);
    }
}
