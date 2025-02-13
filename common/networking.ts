import type { ZodSchema } from "zod";
import { HTTPException } from "hono/http-exception";
import { ContentfulStatusCode } from "hono/utils/http-status";
import { resolver } from "hono-openapi/zod";

import { logger } from "@/common/logging.ts";

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

// Return an HTTP exception for static request errors
export function toHttpException(error: unknown, status = 422) {
    const statusCode = status as ContentfulStatusCode;

    if (error instanceof Error) {
        throw new HTTPException(statusCode, {
            message: error.message,
        });
    }

    throw new HTTPException(statusCode, {
        message: "An unexpected error occurred",
    });
}

// Return an error payload for stream generators
export function toGeneratorError(error: unknown) {
    if (error instanceof Error) {
        logger.error(error.stack ?? error.message);

        return {
            error: {
                message: error.message,
            },
        };
    }

    return {
        error: {
            message: "An unexpected error occurred",
        },
    };
}
