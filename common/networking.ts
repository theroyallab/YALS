import type { ZodSchema } from "zod";
import { resolver } from "hono-openapi/zod";

// Originally from Stoker adopted for hono-openapi
export const jsonContent = <T extends ZodSchema>(
    schema: T,
    description: string,
) => {
    return {
        content: {
            "application/json": {
                schema: resolver(schema),
            },
        },
        description,
    };
};
