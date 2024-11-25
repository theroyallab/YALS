import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { describeRoute } from "hono-openapi";
import { validator as zValidator } from "hono-openapi/zod";
import { jsonContent } from "@/common/networking.ts";

import { CompletionRequest, CompletionResponse } from "./types/completions.ts";
import checkModelMiddleware from "../middleware/checkModelMiddleware.ts";
import { generateCompletion, streamCompletion } from "./utils/completion.ts";
import { logger } from "@/common/logging.ts";

const router = new Hono();

const completionsRoute = describeRoute({
    responses: {
        200: jsonContent(CompletionResponse, "Response to completions"),
    },
});

router.post(
    "/v1/completions",
    completionsRoute,
    checkModelMiddleware,
    zValidator("json", CompletionRequest),
    async (c) => {
        const params = c.req.valid("json");

        if (params.stream) {
            return streamSSE(c, async (stream) => {
                await streamCompletion(stream, c.var.model, params);
            });
        } else {
            let finished = false;
            const abortController = new AbortController();

            c.req.raw.signal.addEventListener("abort", () => {
                if (!finished) {
                    abortController.abort();
                    logger.error("Aborted generation request");
                }
            });

            const completionResult = await generateCompletion(
                c.var.model,
                params,
                abortController.signal,
            );

            finished = true;
            return c.json(completionResult);
        }
    },
);

export default router;
