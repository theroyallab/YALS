import { createRoute, OpenAPIHono } from "@hono/zod-openapi";
import { TestSchema } from "./types/test.ts";
import { defaultHook } from "stoker/openapi";
import { jsonContent } from "stoker/openapi/helpers";

const router = new OpenAPIHono({
    defaultHook: defaultHook,
});

const helloRoute = createRoute({
    method: "get",
    path: "/hello",
    responses: {
        200: jsonContent(TestSchema, "Say hello!"),
    },
});

router.openapi(
    helloRoute,
    (c) => {
        return c.json({
            name: "Hi there!",
        }, 200);
    },
);

export default router;
