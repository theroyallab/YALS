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
            abortController.abort(
                new CancellationError(
                    `Streaming completion ${req.id} cancelled by user.`,
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
    const gen = await staticGenerate(
        req,
        GenerationType.Completion,
        model,
        params.prompt,
        params,
    );
    const response = createResponse(gen, model.path.name);

    return response;
}
