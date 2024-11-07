import { HTTPException } from "hono/http-exception";
import { createMiddleware } from "hono/factory";
import { model } from "@/common/modelContainer.ts";

// Middleware for checking if the model exists
// Sends a validated version of the model via Hono's ctx
const checkModelMiddleware = createMiddleware(async (c, next) => {
    if (!model) {
        throw new HTTPException(401, { message: "A model is not loaded." });
    }

    // Validated reference
    c.set("model", model);

    await next();
});

export default checkModelMiddleware;
