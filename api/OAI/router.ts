import { HTTPException } from "hono/http-exception";
import { createRoute, OpenAPIHono } from "@hono/zod-openapi";
import { defaultHook } from "stoker/openapi";
import { jsonContent, jsonContentRequired } from "stoker/openapi/helpers";
import {
    CompletionRequest,
    CompletionRespChoice,
    CompletionResponse,
} from "./types/completions.ts";
import { model } from "../../common/model.ts";

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
    responses: {
        200: jsonContent(CompletionResponse, "Response to completions"),
    },
});

router.openapi(
    completionsRoute,
    async (c) => {
        if (!model) {
            throw new HTTPException(401, { message: "Model is not loaded!" });
        }

        const params = c.req.valid("json");
        const result = await model.generate(params.prompt) ?? "";
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
