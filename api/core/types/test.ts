import "zod-openapi/extend";
import { z } from "zod";

export const TestSchema = z.object({
    name: z.string().openapi({
        description: "Test string",
        example: "Hello!",
    }),
}).openapi({ description: "Test Schema" });
