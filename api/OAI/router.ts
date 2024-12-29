import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { describeRoute } from "hono-openapi";
import { validator as zValidator } from "hono-openapi/zod";
import { AuthKeyPermission } from "@/common/auth.ts";
import { jsonContent } from "@/common/networking.ts";

import authMiddleware from "@/api/middleware/authMiddleware.ts";
import checkModelMiddleware from "../middleware/checkModelMiddleware.ts";
import { CompletionRequest, CompletionResponse } from "./types/completions.ts";
import { generateCompletion, streamCompletion } from "./utils/completion.ts";

const router = new Hono();

const completionsRoute = describeRoute({
    responses: {
        200: jsonContent(CompletionResponse, "Response to completions"),
    },
});

router.post(
    "/v1/completions",
    completionsRoute,
    authMiddleware(AuthKeyPermission.API),
    checkModelMiddleware,
    zValidator("json", CompletionRequest),
    async (c) => {
        const params = c.req.valid("json");

        if (params.stream) {
            return streamSSE(c, async (stream) => {
                await streamCompletion(c.req, stream, c.var.model, params);
            });
        } else {
            const completionResult = await generateCompletion(
                c.var.model,
                params,
                c.req,
            );

            return c.json(completionResult);
        }
    },
);

export default router;
