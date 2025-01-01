import * as z from "@/common/myZod.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";

export const CompletionResponseFormat = z.object({
    type: z.string().default("text"),
});

export const CommonCompletionRequest = z.object({
    model: z.string().nullish(),
    stream: z.boolean().nullish().coalesce(false),
    logprobs: z.number().gte(0).nullish().coalesce(0),
    response_format: CompletionResponseFormat.nullish().coalesce(
        CompletionResponseFormat.parse({}),
    ),
    n: z.number().gte(1).nullish().coalesce(1),
    best_of: z.number().nullish(),
    echo: z.boolean().nullish().coalesce(false),
    suffix: z.string().nullish(),
    user: z.string().nullish(),
});

export const CompletionRequest = z.object({
    prompt: z.union([
        z.string(),
        z.array(z.string()).transform((arr) => arr.join("\n")),
    ]),
})
    .merge(CommonCompletionRequest)
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
