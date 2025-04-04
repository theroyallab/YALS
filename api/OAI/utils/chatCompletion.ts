import { HonoRequest } from "hono";
import { SSEStreamingApi } from "hono/streaming";

import {
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
        addBosToken = true,
        banEosToken = false,
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

    const bosToken = model.tokenizer.bosToken;
    let prompt = promptTemplate.render({
        ...templateVars,
        messages: messages,
        bos_token: addBosToken ? bosToken?.piece : "",
        eos_token: banEosToken ? "" : model.tokenizer.eosToken?.piece,
        add_generation_prompt: addGenerationPrompt,
    });

    // Remove extra BOS token at start of prompt if present
    // Better to do this since a template can add BOS anywhere
    if (bosToken && prompt.startsWith(bosToken.piece)) {
        prompt = prompt.slice(bosToken.piece.length);
    }

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
    req: HonoRequest,
    stream: SSEStreamingApi,
    model: Model,
    promptTemplate: PromptTemplate,
    params: ChatCompletionRequest,
) {
    const cmplId = `chatcmpl-${crypto.randomUUID().replaceAll("-", "")}`;
    const abortController = new AbortController();
    let finished = false;

    // If an abort happens before streaming starts
    req.raw.signal.addEventListener("abort", () => {
        if (!finished) {
            abortController.abort(
                new CancellationError(
                    `Streaming chat completion ${req.id} cancelled by user.`,
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
            addBosToken: params.add_bos_token,
            banEosToken: params.ban_eos_token,
            addGenerationPrompt: params.add_generation_prompt,
            templateVars: params.template_vars,
        },
    );

    addTemplateMetadata(promptTemplate, params);

    try {
        const generator = model.generateGen(
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
    } catch (error) {
        await stream.writeSSE({
            data: JSON.stringify(toGeneratorError(error)),
        });
    }

    finished = true;
}

export async function generateChatCompletion(
    req: HonoRequest,
    model: Model,
    promptTemplate: PromptTemplate,
    params: ChatCompletionRequest,
) {
    const prompt = applyChatTemplate(
        model,
        promptTemplate,
        params.messages,
        {
            addBosToken: params.add_bos_token,
            banEosToken: params.ban_eos_token,
            addGenerationPrompt: params.add_generation_prompt,
            templateVars: params.template_vars,
        },
    );

    addTemplateMetadata(promptTemplate, params);

    const gen = await staticGenerate(
        req,
        GenerationType.ChatCompletion,
        model,
        prompt,
        params,
    );
    const response = createResponse(gen, model.path.name);

    return response;
}
