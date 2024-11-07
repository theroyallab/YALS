import { HTTPException } from "hono/http-exception";
import { createRoute, OpenAPIHono } from "@hono/zod-openapi";
import { defaultHook } from "stoker/openapi";
import { model } from "../../common/model.ts";

const router = new OpenAPIHono({
    defaultHook: defaultHook,
});

const unloadRoute = createRoute({
    method: "get",
    path: "/v1/model/unload",
    responses: {
        200: {
            description: "Model successfully unloaded"
        }
    },
});

router.openapi(
    unloadRoute,
    async (c) => {
        if (!model) {
            throw new HTTPException(401, { message: "Model is not loaded!" });
        }

        await model.unload();

        c.status(200);
        return c.body(null);
    },
);

export default router;
