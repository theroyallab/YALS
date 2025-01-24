import * as z from "@/common/myZod.ts";
import { ChatCompletionMessage } from "@/api/OAI/types/chatCompletions.ts";

const CommonTokenRequest = z.object({
    add_bos_token: z.boolean().nullish().coalesce(true),
    encode_special_tokens: z.boolean().nullish().coalesce(true),
    decode_special_tokens: z.boolean().nullish().coalesce(true),
});

export const TokenEncodeRequest = z.object({
    text: z.union([z.string(), z.array(ChatCompletionMessage)]),
})
    .merge(CommonTokenRequest);

export const TokenEncodeResponse = z.object({
    tokens: z.array(z.number()),
    length: z.number(),
});

export const TokenDecodeRequest = z.object({
    tokens: z.array(z.number()),
})
    .merge(CommonTokenRequest);

export const TokenDecodeResponse = z.object({
    text: z.string(),
});
