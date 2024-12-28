import { resolver } from "hono-openapi/valibot";
import { BaseIssue, BaseSchema } from "valibot";

// Originally from Stoker adopted for hono-openapi

export const jsonContent = <
    T extends BaseSchema<unknown, unknown, BaseIssue<unknown>>,
>(
    schema: T,
    description: string,
) => {
    return {
        content: {
            "application/json": {
                schema: resolver(schema, { errorMode: "warn" }),
            },
        },
        description,
    };
};
