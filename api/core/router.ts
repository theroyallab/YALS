import { Hono } from "hono";
import { describeRoute } from "hono-openapi";

import checkModelMiddleware from "../middleware/checkModelMiddleware.ts";

const router = new Hono();

const unloadRoute = describeRoute({
    responses: {
        200: {
            description: "Model successfully unloaded",
        },
    },
});

router.post(
    unloadRoute,
    checkModelMiddleware,
    async (c) => {
        await c.var.model.unload();

        c.status(200);
        return c.body(null);
    },
);

export default router;
