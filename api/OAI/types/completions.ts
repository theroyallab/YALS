import "zod-openapi/extend";
import { z } from "zod";
import { BaseSamplerRequest } from "@/common/sampling.ts";

export const CompletionResponseFormat = z.object({
    type: z.string().default("text"),
});

export const CompletionRequest = z.object({
    model: z.string().optional(),
    prompt: z.string(),
    stream: z.boolean().default(false),
    logprobs: z.number().gte(0).optional().default(0),
    response_format: CompletionResponseFormat.optional(),
    n: z.number().gte(1).optional().default(1),
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
    id: z.string().default(crypto.randomUUID().replaceAll("-", "")),
    choices: z.array(CompletionRespChoice),
    created: z.number().default((new Date()).getSeconds()),
    model: z.string(),
    object: z.string().default("text_completion"),
});
