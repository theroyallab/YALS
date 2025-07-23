import * as z from "@/common/myZod.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";

export const CompletionResponseFormat = z.object({
    type: z.string().default("text"),
});

export const UsageStats = z.object({
    prompt_tokens: z.number(),
    prompt_time: z.number().optional(),
    prompt_tokens_per_sec: z.number().optional(),
    completion_tokens: z.number(),
    completion_time: z.number().optional(),
    completion_tokens_per_sec: z.number().optional(),
    total_tokens: z.number(),
    total_time: z.number().optional(),
});

export type UsageStats = z.infer<typeof UsageStats>;

export const CommonCompletionRequest = z.object({
    model: z.string().cleanOptional(),
    stream: z.boolean().nullish().coalesce(false),
    logprobs: z.number().gte(0).nullish().coalesce(0),
    response_format: CompletionResponseFormat.nullish().coalesce(
        CompletionResponseFormat.parse({}),
    ),
    n: z.number().gte(1).nullish().coalesce(1),
    best_of: z.number().cleanOptional(),
    echo: z.boolean().nullish().coalesce(false),
    suffix: z.string().cleanOptional(),
    user: z.string().cleanOptional(),
})
    .and(BaseSamplerRequest)
    .transform((obj) => {
        if (obj.response_format.type === "json") {
            obj.json_schema = {
                "type": "object",
            };
        }

        return obj;
    });

export type CommonCompletionRequest = z.infer<typeof CommonCompletionRequest>;

export const CompletionRequest = z.object({
    prompt: z.union([
        z.string(),
        z.array(z.string()).transform((arr) => arr.join("\n")),
    ]),
})
    .and(CommonCompletionRequest)
    .describe("Completion Request parameters");

export type CompletionRequest = z.infer<typeof CompletionRequest>;

export const CompletionRespChoice = z.object({
    index: z.number().default(0),
    finish_reason: z.string().optional(),
    text: z.string(),
});

export type CompletionRespChoice = z.infer<typeof CompletionRespChoice>;

export const CompletionResponse = z.object({
    id: z.string().default(`cmpl-${crypto.randomUUID().replaceAll("-", "")}`),
    choices: z.array(CompletionRespChoice),
    created: z.number().default(Math.floor(Date.now() / 1000)),
    model: z.string(),
    object: z.string().default("text_completion"),
    usage: UsageStats.optional(),
});
