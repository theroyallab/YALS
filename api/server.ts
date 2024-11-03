import { Hono } from "hono";
import core from "./core/router.ts"

export function createApi() {
    const app = new Hono();

    app.route("/", core);

    Deno.serve({
        hostname: "127.0.0.1",
        port: 5000,
        handler: app.fetch 
    });
}
