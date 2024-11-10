import { logger as loggerMiddleware } from "hono/logger";
import { cors } from "hono/cors";
import { getLogger } from "logtape";
import { apiReference } from "@scalar/hono-api-reference";
import { notFound, onError } from "stoker/middlewares";

import { Hono } from "hono";
import { openAPISpecs } from "hono-openapi";

import core from "./core/router.ts";
import oai from "./OAI/router.ts";

const logger = getLogger("YALS");

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

    // TODO: Create custom handlers
    app.onError(onError);
    app.notFound(notFound);

    // Serve
    Deno.serve({
        hostname: "127.0.0.1",
        port: 5000,
        handler: app.fetch,
        onListen: ({ hostname, port }) => {
            logger.info(`Server running on http://${hostname}:${port}`);
        },
    });
}
