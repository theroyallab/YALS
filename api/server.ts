import { Hono } from "hono";
import { logger } from "hono/logger";
import { cors } from "hono/cors";
import core from "./core/router.ts";

export function createApi() {
    const app = new Hono();

    // Middleware
    app.use(logger());
    app.use("*", cors());

    // Add routers
    app.route("/", core);

    // Serve
    Deno.serve({
        hostname: "127.0.0.1",
        port: 5000,
        handler: app.fetch,
    });
}
