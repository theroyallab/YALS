import * as z from "@/common/myZod.ts";
import {
    CommonCompletionRequest,
    UsageStats,
} from "@/api/OAI/types/completions.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";

const ChatCompletionImageUrl = z.object({
    url: z.string(),
});

const ChatCompletionMessagePart = z.object({
    type: z.string().nullish().coalesce("text").openapi({
        effectType: "input",
    }),
    text: z.string().nullish(),
    image_url: ChatCompletionImageUrl.nullish(),
});

export type ChatCompletionMessagePart = z.infer<
    typeof ChatCompletionMessagePart
>;

export const ChatCompletionMessage = z.object({
    role: z.string().default("user"),
    content: z.union([z.string(), z.array(ChatCompletionMessagePart)]),
});

export type ChatCompletionMessage = z.infer<typeof ChatCompletionMessage>;

const ChatCompletionResponseFormat = z.object({
    type: z.string().default("text"),
});

const ChatCompletionStreamOptions = z.object({
    include_usage: z.boolean().nullish().coalesce(false),
});

// TODO: Merge together with Zod 4
const ChatCompletionBaseRequest = z.object({
    messages: z.array(ChatCompletionMessage).nullish().coalesce([]),
    response_format: ChatCompletionResponseFormat.nullish().coalesce(
        ChatCompletionResponseFormat.parse({}),
    ),
    stream_options: ChatCompletionStreamOptions.nullish(),
    add_generation_prompt: z.boolean().nullish().coalesce(true),
    prompt_template: z.string().nullish(),
    template_vars: z.record(z.unknown()).nullish().coalesce({}),
})
    .merge(CommonCompletionRequest);

export const ChatCompletionRequest = z.aliasedObject(
    ChatCompletionBaseRequest,
    [
        { field: "template_vars", aliases: ["chat_template_kwargs"] },
    ],
)
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
    delta: z.union([ChatCompletionMessage, z.record(z.unknown())]),
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
