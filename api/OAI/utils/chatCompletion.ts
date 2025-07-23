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
import { ToolSpec } from "../types/tools.ts";
import { TOOL_CALL_SCHEMA, ToolCallProcessor } from "./tools.ts";

interface TemplateFormatOptions {
    addBosToken?: boolean;
    banEosToken?: boolean;
    addGenerationPrompt?: boolean;
    templateVars?: Record<string, unknown>;
    tools?: ToolSpec[];
    responsePrefix?: string;
}

function createResponse(chunks: FinishChunk[], modelName: string) {
    const choices: ChatCompletionRespChoice[] = [];

    for (const chunk of chunks) {
        const message = ChatCompletionMessage.parse({
            role: "assistant",
            content: chunk.text,
        });

        if (chunk.toolCalls) {
            message.tool_calls = ToolCallProcessor.fromJson(chunk.toolCalls);
        }

        const choice = ChatCompletionRespChoice.parse({
            index: chunk.taskIdx,
            message: message,
            finish_reason: convertFinishReason(chunk),
        });

        choices.push(choice);
    }

    const finalChunk = chunks.at(-1);
    const usage = finalChunk ? createUsageStats(finalChunk) : undefined;

    const response = ChatCompletionResponse.parse({
        choices: choices,
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

    if (chunk.kind === "finish" && chunk.toolCalls) {
        message.tool_calls = ToolCallProcessor.fromJson(chunk.toolCalls);
    }

    const choice = ChatCompletionStreamChoice.parse({
        index: chunk.taskIdx,
        delta: message,
    });

    if (chunk.kind === "finish") {
        choice.finish_reason = convertFinishReason(chunk);
    }

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

    const bosToken = model.tokenizer.bosToken;
    let prompt = promptTemplate.render({
        ...templateVars,
        messages: messages,
        bos_token: bosToken?.piece ?? "",
        eos_token: model.tokenizer.eosToken?.piece ?? "",
        add_generation_prompt: addGenerationPrompt,
        tools: options.tools ?? null,
    });

    if (options.responsePrefix) {
        if (addGenerationPrompt) {
            prompt += options.responsePrefix;
        } else {
            logger.warn(
                "Could not add response prefix because " +
                    "add_generation_prompt is False",
            );
        }
    }

    // Remove extra BOS token at start of prompt if present
    // Some model templates don't respect their own add_bos_token setting
    // Better to do this since a template can add BOS anywhere
    if (
        bosToken && model.tokenizer.addBosToken &&
        prompt.startsWith(bosToken.piece)
    ) {
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

    if (metadata.tool_start) {
        params.stop.push(metadata.tool_start);
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

    const toolStart = promptTemplate.metadata.tool_start;
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
            tools: params.tools,
            responsePrefix: params.response_prefix,
        },
    );

    addTemplateMetadata(promptTemplate, params);

    try {
        const queue = new Queue<GenerationChunk | Error>();
        const genTasks = [];

        for (let i = 0; i < params.n; i++) {
            const task = streamCollector(
                requestId,
                prompt,
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

            if (chunk.kind === "finish") {
                // Handle tools
                if (toolStart && chunk.stopToken) {
                    await generateToolCalls(
                        requestId,
                        prompt,
                        [chunk],
                        params,
                        model,
                        promptTemplate,
                        requestSignal,
                    );
                }

                completedTasks++;
            }

            const streamChunk = createStreamChunk(
                chunk,
                model.path.name,
                cmplId,
            );
            await stream.writeSSE({ data: JSON.stringify(streamChunk) });

            // TODO: Make usage aggregated
            if (completedTasks === params.n && queue.size === 0) {
                if (
                    params.stream_options?.include_usage &&
                    chunk.kind === "finish"
                ) {
                    const usageChunk = createUsageChunk(
                        chunk,
                        model.path.name,
                        cmplId,
                    );

                    await stream.writeSSE({ data: JSON.stringify(usageChunk) });
                }

                logger.info(
                    `Finished streaming chat completion request ${requestId}`,
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
            tools: params.tools,
            responsePrefix: params.response_prefix,
        },
    );

    addTemplateMetadata(promptTemplate, params);

    // Handle generation in the common function
    const generations = await staticGenerate(
        requestId,
        GenerationType.ChatCompletion,
        prompt,
        params,
        model,
        requestSignal,
    );

    // Check for tool calls
    await generateToolCalls(
        requestId,
        prompt,
        generations,
        params,
        model,
        promptTemplate,
        requestSignal,
    );

    const response = createResponse(generations, model.path.name);

    logger.info(`Finished chat completion request ${requestId}`);
    return response;
}

async function generateToolCalls(
    requestId: string,
    prompt: string,
    gens: FinishChunk[],
    params: ChatCompletionRequest,
    model: Model,
    promptTemplate: PromptTemplate,
    requestSignal: AbortSignal,
) {
    const toolGenTasks = [];
    const toolStart = promptTemplate.metadata.tool_start;
    if (!toolStart) {
        return;
    }

    const toolIdx = [];

    const toolParams = structuredClone(params);
    toolParams.json_schema = TOOL_CALL_SCHEMA;

    for (const [index, gen] of gens.entries()) {
        if (!gen.stopToken.startsWith(toolStart)) {
            continue;
        }

        const genRequestId = params.n > 1
            ? `${requestId}-${gen.taskIdx}`
            : requestId;
        logger.info(`Tool call detected for request ${genRequestId}`);

        if (gen.fullText) {
            prompt += prompt + gen.fullText;
        }

        const toolRequestid = `${genRequestId}-tool`;
        const toolTask = staticGenerate(
            toolRequestid,
            GenerationType.ChatCompletion,
            prompt,
            toolParams,
            model,
            requestSignal,
        );

        toolGenTasks.push(toolTask);
        toolIdx.push(index);
    }

    if (toolIdx.length > 0) {
        const toolGenResults = await Promise.allSettled(toolGenTasks);

        for (const [i, genIdx] of toolIdx.entries()) {
            const toolResult = toolGenResults[i];
            if (toolResult.status === "fulfilled" && toolResult.value[0]) {
                const toolGen = toolResult.value[0];
                gens[genIdx].toolCalls = toolGen.text;
            }
        }
    }
}
