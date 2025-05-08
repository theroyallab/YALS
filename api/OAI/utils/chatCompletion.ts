import { SSEStreamingApi } from "hono/streaming";

import {
    convertFinishReason,
    createUsageStats,
    GenerationType,
    staticGenerate,
} from "@/api/OAI/utils/generation.ts";
import { Model } from "@/bindings/bindings.ts";
import { FinishChunk, GenerationChunk } from "@/bindings/types.ts";
import { toGeneratorError } from "@/common/networking.ts";
import { PromptTemplate } from "@/common/templating.ts";

import {
    ChatCompletionMessage,
    ChatCompletionMessagePart,
    ChatCompletionRequest,
    ChatCompletionRespChoice,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
} from "../types/chatCompletions.ts";
import { CancellationError } from "@/common/errors.ts";
import { logger } from "@/common/logging.ts";

interface TemplateFormatOptions {
    addBosToken?: boolean;
    banEosToken?: boolean;
    addGenerationPrompt?: boolean;
    templateVars?: Record<string, unknown>;
}

function createResponse(chunk: FinishChunk, modelName: string) {
    const message = ChatCompletionMessage.parse({
        role: "assistant",
        content: chunk.text,
    });

    const choice = ChatCompletionRespChoice.parse({
        message: message,
        finish_reason: convertFinishReason(chunk),
    });

    const usage = createUsageStats(chunk);

    const response = ChatCompletionResponse.parse({
        choices: [choice],
        model: modelName,
        usage,
    });

    return response;
}

function createStreamChunk(
    chunk: GenerationChunk,
    modelName: string,
    cmplId: string,
) {
    const message = ChatCompletionMessage.parse({
        role: "assistant",
        content: chunk.text,
    });

    const choice = ChatCompletionStreamChoice.parse({
        delta: message,
    });

    const response = ChatCompletionStreamChunk.parse({
        id: cmplId,
        choices: [choice],
        model: modelName,
    });

    return response;
}

function createUsageChunk(
    chunk: FinishChunk,
    modelName: string,
    cmplId: string,
) {
    const response = ChatCompletionStreamChunk.parse({
        id: cmplId,
        model: modelName,
        usage: createUsageStats(chunk),
    });

    return response;
}

export function applyChatTemplate(
    model: Model,
    promptTemplate: PromptTemplate,
    messages: ChatCompletionMessage[],
    options: TemplateFormatOptions = {},
): string {
    const {
        addGenerationPrompt = true,
        templateVars = {},
    } = options;

    messages.forEach((message) => {
        if (Array.isArray(message.content)) {
            const messageParts = message.content as ChatCompletionMessagePart[];
            message.content = messageParts.find((part) =>
                part.type === "text"
            )?.text ?? "";
        }
    });

    const prompt = promptTemplate.render({
        ...templateVars,
        messages: messages,
        bos_token: model.tokenizer.bosToken?.piece ?? "",
        eos_token: model.tokenizer.eosToken?.piece ?? "",
        add_generation_prompt: addGenerationPrompt,
    });

    return prompt;
}

function addTemplateMetadata(
    promptTemplate: PromptTemplate,
    params: ChatCompletionRequest,
) {
    const metadata = promptTemplate.metadata;

    if (metadata.stop_strings) {
        params.stop.push(...metadata.stop_strings);
    }
}

// TODO: Possibly rewrite this to unify with completions
export async function streamChatCompletion(
    requestId: string,
    stream: SSEStreamingApi,
    params: ChatCompletionRequest,
    model: Model,
    promptTemplate: PromptTemplate,
    requestSignal: AbortSignal,
) {
    logger.info(`Received streaming chat completion request ${requestId}`);

    const cmplId = `chatcmpl-${crypto.randomUUID().replaceAll("-", "")}`;
    const abortController = new AbortController();
    let finished = false;

    // If an abort happens before streaming starts
    requestSignal.addEventListener("abort", () => {
        if (!finished) {
            abortController.abort(
                new CancellationError(
                    `Streaming chat completion ${requestId} cancelled by user.`,
                ),
            );
            finished = true;
        }
    });

    const prompt = applyChatTemplate(
        model,
        promptTemplate,
        params.messages,
        {
            addGenerationPrompt: params.add_generation_prompt,
            templateVars: params.template_vars,
        },
    );

    addTemplateMetadata(promptTemplate, params);

    try {
        const generator = model.generateGen(
            requestId,
            prompt,
            params,
            abortController.signal,
        );

        for await (const chunk of generator) {
            const streamChunk = createStreamChunk(
                chunk,
                model.path.name,
                cmplId,
            );

            await stream.writeSSE({
                data: JSON.stringify(streamChunk),
            });

            // Write usage stats if user requests it
            if (
                params.stream_options?.include_usage && chunk.kind === "finish"
            ) {
                const usageChunk = createUsageChunk(
                    chunk,
                    model.path.name,
                    cmplId,
                );

                await stream.writeSSE({
                    data: JSON.stringify(usageChunk),
                });
            }
        }

        logger.info(`Finished streaming chat completion request ${requestId}`);
    } catch (error) {
        await stream.writeSSE({
            data: JSON.stringify(toGeneratorError(error)),
        });
    }

    finished = true;
}

export async function generateChatCompletion(
    requestId: string,
    params: ChatCompletionRequest,
    model: Model,
    promptTemplate: PromptTemplate,
    requestSignal: AbortSignal,
) {
    logger.info(`Received chat completion request ${requestId}`);

    const prompt = applyChatTemplate(
        model,
        promptTemplate,
        params.messages,
        {
            addGenerationPrompt: params.add_generation_prompt,
            templateVars: params.template_vars,
        },
    );

    addTemplateMetadata(promptTemplate, params);

    // Handle generation in the common function
    const gen = await staticGenerate(
        requestId,
        GenerationType.ChatCompletion,
        prompt,
        params,
        model,
        requestSignal,
    );
    const response = createResponse(gen, model.path.name);

    return response;
}
