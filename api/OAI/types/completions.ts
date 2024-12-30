import * as z from "@/common/myZod.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";

export const CompletionResponseFormat = z.object({
    type: z.string().default("text"),
});

export const CompletionRequest = z.object({
    model: z.string().nullish(),
    prompt: z.string(),
    stream: z.boolean().nullish().coalesce(false),
    logprobs: z.number().gte(0).nullish().coalesce(0),
    response_format: CompletionResponseFormat.nullish(),
    n: z.number().gte(1).nullish().coalesce(1),
})
    .and(BaseSamplerRequest)
    .openapi({
        description: "Completion Request parameters",
    });

export type CompletionRequest = z.infer<typeof CompletionRequest>;

export const CompletionRespChoice = z.object({
    index: z.number().default(0),
    finish_reason: z.string().optional(),
    text: z.string(),
});

export const CompletionResponse = z.object({
    id: z.string().default(`cmpl-${crypto.randomUUID().replaceAll("-", "")}`),
    choices: z.array(CompletionRespChoice),
    created: z.number().default((new Date()).getSeconds()),
    model: z.string(),
    object: z.string().default("text_completion"),
});
