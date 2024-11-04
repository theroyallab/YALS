import { z } from "@hono/zod-openapi";

export const CompletionsRequest = z.object({
    prompt: z.string(),
});

export const CompletionsResponse = z.object({
    result: z.string(),
});
