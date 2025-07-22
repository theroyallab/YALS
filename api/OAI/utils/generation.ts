import {
    CommonCompletionRequest,
    UsageStats,
} from "@/api/OAI/types/completions.ts";
import { Model } from "@/bindings/bindings.ts";
import { FinishChunk, GenerationChunk } from "@/bindings/types.ts";
import { toHttpException } from "@/common/networking.ts";
import { CancellationError } from "@/common/errors.ts";
import { Queue } from "@core/asyncutil/queue";

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
    requestId: string,
    genType: GenerationType,
    prompt: string,
    params: CommonCompletionRequest,
    model: Model,
    requestSignal: AbortSignal,
) {
    const abortController = new AbortController();
    let finished = false;

    requestSignal.addEventListener("abort", () => {
        if (!finished) {
            abortController.abort(
                new CancellationError(
                    `${genType} ${requestId} cancelled by user.`,
                ),
            );
            finished = true;
        }
    });

    try {
        const genTasks: Promise<FinishChunk>[] = [];

        for (let i = 0; i < params.n; i++) {
            const genRequestId = params.n > 1 ? `${requestId}-${i}` : requestId;
            console.log(i);
            const task = model.generate(
                genRequestId,
                prompt,
                params,
                requestSignal,
                i,
            );

            genTasks.push(task);
        }

        const genResults = await Promise.allSettled(genTasks);
        const generations = genResults.reduce((acc, result) => {
            if (result.status === "rejected") {
                return acc;
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
    requestId: string,
    prompt: string,
    params: CommonCompletionRequest,
    model: Model,
    requestSignal: AbortSignal,
    taskIdx: number,
    genQueue: Queue<GenerationChunk | Error>,
) {
    try {
        const genRequestId = params.n > 1
            ? `${requestId}-${taskIdx}`
            : requestId;
        const generator = model.generateGen(
            genRequestId,
            prompt,
            params,
            requestSignal,
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
