import { createMiddleware } from "hono/factory";

import { Model } from "@/bindings/bindings.ts";
import { OAIContext } from "../OAI/types/context.ts";

// Extra vars for context
interface CtxOptions {
    Variables: {
        model: Model;
        oaiCtx: OAIContext;
    };
}

// Middleware to create an OAI ctx object
const oaiContextMiddleware = createMiddleware<CtxOptions>(
    async (c, next) => {
        const ctx: OAIContext = {
            requestId: c.var.requestId,
            model: c.var.model,
            cancellationSignal: c.req.raw.signal,
        };

        c.set("oaiCtx", ctx);

        await next();
    },
);

export default oaiContextMiddleware;
