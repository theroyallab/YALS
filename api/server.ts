import { cors } from "hono/cors";
import { logger as loggerMiddleware } from "hono/logger";
import { ContentfulStatusCode } from "hono/utils/http-status";
import { apiReference } from "@scalar/hono-api-reference";

import { Hono } from "hono";
import { openAPISpecs } from "hono-openapi";

import core from "./core/router.ts";
import oai from "./OAI/router.ts";
import { config } from "@/common/config.ts";
import { logger } from "@/common/logging.ts";

export function createApi() {
    const app = new Hono();

    // TODO: Use a custom middleware instead of overriding Hono's logger
    const printToLogtape = (message: string, ...rest: string[]) => {
        logger.info(message, { rest });
    };

    // Middleware
    app.use(loggerMiddleware(printToLogtape));
    app.use("*", cors());

    // Add routers
    app.route("/", core);
    app.route("/", oai);

    // OpenAPI documentation
    app.get(
        "/openapi.json",
        openAPISpecs(app, {
            documentation: {
                openapi: "3.0.0",
                info: {
                    version: "0.0.1",
                    title: "YALS",
                },
            },
        }),
    );

    app.get(
        "/reference",
        apiReference({
            spec: {
                url: "/openapi.json",
            },
        }),
    );

    // Error handling
    // Originally from the Stoker package
    app.onError((err, c) => {
        const currentStatus = "status" in err
            ? err.status
            : c.newResponse(null).status;
        const statusCode = currentStatus != 200
            ? (currentStatus as ContentfulStatusCode)
            : 500;

        // Don't log a traceback for unauthorized
        if (err.stack && currentStatus != 401) {
            logger.error(err.stack);
        }

        return c.json({
            detail: err.message,
        }, statusCode);
    });

    app.notFound((c) => {
        return c.json({
            message: `Method or path not found - ${c.req.method} ${c.req.path}`,
        }, 404);
    });

    // Serve
    Deno.serve({
        hostname: config.network.host,
        port: config.network.port,
        handler: app.fetch,
        onListen: ({ hostname, port }) => {
            logger.info(`Server running on http://${hostname}:${port}`);
        },
    });
}
