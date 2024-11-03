import { Hono } from "hono";
import { logger as loggerMiddleware } from "hono/logger";
import { cors } from "hono/cors";
import { getLogger } from "logtape";
import core from "./core/router.ts";

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
