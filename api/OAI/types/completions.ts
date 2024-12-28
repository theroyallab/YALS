import * as v from "valibot";
import { BaseSamplerRequest } from "@/common/sampling.ts";

export const CompletionResponseFormat = v.object({
    type: v.nullish(v.string(), "text"),
});

export const CompletionRequest = v.intersect([
    v.object({
        model: v.nullish(v.string()),
        prompt: v.string(),
        stream: v.nullish(v.boolean(), false),
        logprobs: v.nullish(v.pipe(v.number(), v.minValue(0)), 0),
        response_format: v.nullish(CompletionResponseFormat),
        n: v.nullish(v.pipe(v.number(), v.minValue(1)), 1),
    }),
    BaseSamplerRequest,
]);
// .openapi({
//     description: "Completion Request parameters",
// });

export type CompletionRequest = v.InferOutput<typeof CompletionRequest>;

export const CompletionRespChoice = v.object({
    index: v.nullish(v.number(), 0),
    finish_reason: v.nullish(v.string()),
    text: v.string(),
});

export const CompletionResponse = v.object({
    id: v.nullish(v.string(), crypto.randomUUID().replaceAll("-", "")),
    choices: v.array(CompletionRespChoice),
    created: v.nullish(v.number(), (new Date()).getSeconds()),
    model: v.string(),
    object: v.nullish(v.string(), "text_completion"),
});
