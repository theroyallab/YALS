import { createRoute, OpenAPIHono } from "@hono/zod-openapi";
import { defaultHook } from "stoker/openapi";
import { jsonContent, jsonContentRequired } from "stoker/openapi/helpers";

import {
    CompletionRequest,
    CompletionRespChoice,
    CompletionResponse,
} from "./types/completions.ts";
import checkModelMiddleware from "../middleware/checkModelMiddleware.ts";

const router = new OpenAPIHono({
    defaultHook: defaultHook,
});

const completionsRoute = createRoute({
    method: "post",
    path: "/v1/completions",
    request: {
        body: jsonContentRequired(
            CompletionRequest,
            "Request for a completion",
        ),
    },
    middleware: [checkModelMiddleware],
    responses: {
        200: jsonContent(CompletionResponse, "Response to completions"),
    },
});

router.openapi(
    completionsRoute,
    async (c) => {
        const params = c.req.valid("json");
        const result = await c.var.model.generate(params.prompt);
        const completionChoice = await CompletionRespChoice.parseAsync({
            text: result,
            index: 0,
        });

        return c.json(
            await CompletionResponse.parseAsync({
                choices: [completionChoice],
                model: "test",
            }),
        );
    },
);

export default router;
