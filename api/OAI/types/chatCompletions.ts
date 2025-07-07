import * as z from "@/common/myZod.ts";
import {
    CommonCompletionRequest,
    UsageStats,
} from "@/api/OAI/types/completions.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";

import { ToolCall, ToolSpec } from "./tools.ts";

const ChatCompletionImageUrl = z.object({
    url: z.string(),
});

const ChatCompletionMessagePart = z.object({
    type: z.string().nullish().coalesce("text"),
    text: z.string().cleanOptional(),
    image_url: ChatCompletionImageUrl.cleanOptional(),
});

export type ChatCompletionMessagePart = z.infer<
    typeof ChatCompletionMessagePart
>;

export const ChatCompletionMessage = z.object({
    role: z.string().default("user"),
    content: z.union([z.string(), z.array(ChatCompletionMessagePart)])
        .cleanOptional(),
    tool_calls: z.array(ToolCall).cleanOptional(),
    tool_call_id: z.string().cleanOptional(),
});

export type ChatCompletionMessage = z.infer<typeof ChatCompletionMessage>;

const ChatCompletionStreamOptions = z.object({
    include_usage: z.boolean().nullish().coalesce(false),
});

export const ChatCompletionRequest = z.aliasedObject(
    z.object({
        messages: z.array(ChatCompletionMessage).nullish().coalesce([]),
        stream_options: ChatCompletionStreamOptions.cleanOptional(),
        add_generation_prompt: z.boolean().nullish().coalesce(true),
        prompt_template: z.string().cleanOptional(),
        template_vars: z.record(z.string(), z.unknown()).nullish().coalesce({}),
        response_prefix: z.string().cleanOptional(),
        tools: z.array(ToolSpec).cleanOptional(),
    }),
    [
        { field: "template_vars", aliases: ["chat_template_kwargs"] },
    ],
)
    .and(CommonCompletionRequest)
    .and(BaseSamplerRequest)
    .transform((obj) => {
        // Always unset add_bos_token
        obj.add_bos_token = undefined;
        return obj;
    });

export type ChatCompletionRequest = z.infer<typeof ChatCompletionRequest>;

export const ChatCompletionRespChoice = z.object({
    index: z.number().default(0),
    finish_reason: z.string().optional(),
    message: ChatCompletionMessage,
});

export const ChatCompletionResponse = z.object({
    id: z.string().default(
        `chatcmpl-${crypto.randomUUID().replaceAll("-", "")}`,
    ),
    choices: z.array(ChatCompletionRespChoice),
    created: z.number().default(Math.floor(Date.now() / 1000)),
    model: z.string(),
    object: z.string().default("chat.completion"),
    usage: UsageStats.optional(),
});

export const ChatCompletionStreamChoice = z.object({
    index: z.number().default(0),
    finish_reason: z.string().optional(),
    delta: z.union([ChatCompletionMessage, z.record(z.string(), z.unknown())]),
});

export const ChatCompletionStreamChunk = z.object({
    id: z.string().default(
        `chatcmpl-${crypto.randomUUID().replaceAll("-", "")}`,
    ),
    choices: z.array(ChatCompletionStreamChoice).default([]),
    created: z.number().default(Math.floor(Date.now() / 1000)),
    model: z.string(),
    object: z.string().default("chat.completion.chunk"),
    usage: UsageStats.optional(),
});
