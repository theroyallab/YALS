import { Hono } from "hono";
import { describeRoute } from "hono-openapi";
import * as modelContainer from "@/common/modelContainer.ts";

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
    "/v1/unload",
    unloadRoute,
    checkModelMiddleware,
    async (c) => {
        await modelContainer.unloadModel();

        c.status(200);
        return c.body(null);
    },
);

export default router;
