import { createRoute, OpenAPIHono } from "@hono/zod-openapi";
import { defaultHook } from "stoker/openapi";

import checkModelMiddleware from "../middleware/checkModelMiddleware.ts";

const router = new OpenAPIHono({
    defaultHook: defaultHook,
});

const unloadRoute = createRoute({
    method: "get",
    path: "/v1/model/unload",
    middleware: [checkModelMiddleware],
    responses: {
        200: {
            description: "Model successfully unloaded",
        },
    },
});

router.openapi(
    unloadRoute,
    async (c) => {
        await c.var.model.unload();

        c.status(200);
        return c.body(null);
    },
);

export default router;
