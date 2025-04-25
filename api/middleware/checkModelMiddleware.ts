import { createMiddleware } from "hono/factory";

import { Model } from "@/bindings/bindings.ts";
import { ModelNotLoadedError } from "@/common/errors.ts";
import { model } from "@/common/modelContainer.ts";

// Extra vars for context
interface CtxOptions {
    Variables: {
        model: Model;
    };
}

// Middleware for checking if the model exists
// Sends a validated version of the model via Hono's ctx
const checkModelMiddleware = createMiddleware<CtxOptions>(
    async (c, next) => {
        if (!model) {
            throw new ModelNotLoadedError();
        }

        // Validated reference
        c.set("model", model);

        await next();
    },
);

export default checkModelMiddleware;
