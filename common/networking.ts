import * as z from "@/common/myZod.ts";
import { HTTPException } from "hono/http-exception";
import { ContentfulStatusCode } from "hono/utils/http-status";
import { resolver } from "hono-openapi";

import { CancellationError } from "@/common/errors.ts";
import { logger } from "@/common/logging.ts";

// Originally from Stoker adopted for hono-openapi
export const jsonContent = <T extends z.ZodType>(
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
    let message = "An unexpected error occurred";

    if (error instanceof CancellationError) {
        status = 408;
        message = error.message;
    } else if (error instanceof Error) {
        message = error.message;
    }

    const statusCode = status as ContentfulStatusCode;
    throw new HTTPException(statusCode, { message });
}

// Return an error payload for stream generators
export function toGeneratorError(error: unknown) {
    let message = "An unexpected error occurred";

    if (error instanceof CancellationError) {
        logger.error(error.message);
        message = error.message;
    } else if (error instanceof Error) {
        logger.error(error.stack || error.message);
        message = error.message;
    }

    return {
        error: {
            message,
        },
    };
}
