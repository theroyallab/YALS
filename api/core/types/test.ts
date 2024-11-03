import { z } from "@hono/zod-openapi";

export const TestSchema = z.object({
    name: z.string().openapi({
        description: "Test string",
        example: "Hello!",
    }),
}).openapi({ description: "Test Schema" });
