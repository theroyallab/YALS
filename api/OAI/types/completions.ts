import { z } from "@hono/zod-openapi";

export const CompletionResponseFormat = z.object({
    type: z.string().default("text"),
});

export const CompletionRequest = z.object({
    model: z.optional(z.string()),
    prompt: z.string(),
    stream: z.boolean().default(false),
    logprobs: z.optional(z.number().gte(0)).default(0),
    response_format: z.optional(CompletionResponseFormat),
    n: z.optional(z.number().gte(1)).default(1),
});

export const CompletionRespChoice = z.object({
    index: z.number().default(0),
    finish_reason: z.optional(z.string()),
    text: z.string(),
});

export const CompletionResponse = z.object({
    id: z.string().default(crypto.randomUUID().replaceAll("-", "")),
    choices: z.array(CompletionRespChoice),
    created: z.number().default((new Date()).getSeconds()),
    model: z.string(),
    object: z.string().default("text_completion"),
});
