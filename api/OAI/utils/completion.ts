import { StreamingApi } from "hono/stream";
import { Model } from "@/bindings/bindings.ts";

import {
    CompletionRequest,
    CompletionRespChoice,
    CompletionResponse,
} from "../types/completions.ts";

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
    stream: StreamingApi,
    model: Model,
    params: CompletionRequest,
) {
    const generator = model.generateGen(params.prompt, params);
    for await (const chunk of generator) {
        const streamChunk = await createResponse(chunk, model.path.name);

        await stream.writeSSE({
            data: JSON.stringify(streamChunk),
        });
    }
}

export async function generateCompletion(
    model: Model,
    params: CompletionRequest,
    abortSignal: AbortSignal,
) {
    const result = await model.generate(params.prompt, params, abortSignal);
    return await createResponse(result, model.path.name);
}
