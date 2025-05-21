import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { streamSSE } from "hono/streaming";
import { describeRoute } from "hono-openapi";
import { validator as sValidator } from "hono-openapi";
import {
    ChatCompletionRequest,
    ChatCompletionResponse,
} from "@/api/OAI/types/chatCompletions.ts";
import {
    generateChatCompletion,
    streamChatCompletion,
} from "@/api/OAI/utils/chatCompletion.ts";
import { AuthKeyPermission } from "@/common/auth.ts";
import { jsonContent } from "@/common/networking.ts";
import { PromptTemplate } from "@/common/templating.ts";

import authMiddleware from "../middleware/authMiddleware.ts";
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
    sValidator("json", CompletionRequest),
    async (c) => {
        const params = c.req.valid("json");

        if (params.stream) {
            return streamSSE(c, async (stream) => {
                await streamCompletion(
                    c.var.requestId,
                    stream,
                    params,
                    c.var.model,
                    c.req.raw.signal,
                );
            });
        } else {
            const completionResult = await generateCompletion(
                c.var.requestId,
                params,
                c.var.model,
                c.req.raw.signal,
            );

            return c.json(completionResult);
        }
    },
);

const chatCompletionsRoute = describeRoute({
    responses: {
        200: jsonContent(
            ChatCompletionResponse,
            "Response to chat completions",
        ),
    },
});

router.post(
    "/v1/chat/completions",
    chatCompletionsRoute,
    authMiddleware(AuthKeyPermission.API),
    checkModelMiddleware,
    sValidator("json", ChatCompletionRequest),
    async (c) => {
        const params = c.req.valid("json");

        let promptTemplate: PromptTemplate;
        if (c.var.model.promptTemplate) {
            promptTemplate = c.var.model.promptTemplate;
        } else {
            throw new HTTPException(422, {
                message:
                    "Chat completions are disabled because a prompt template isn't set.",
            });
        }

        if (params.stream) {
            return streamSSE(c, async (stream) => {
                await streamChatCompletion(
                    c.var.requestId,
                    stream,
                    params,
                    c.var.model,
                    promptTemplate,
                    c.req.raw.signal,
                );
            });
        } else {
            const chatCompletionResult = await generateChatCompletion(
                c.var.requestId,
                params,
                c.var.model,
                promptTemplate,
                c.req.raw.signal,
            );
            return c.json(chatCompletionResult);
        }
    },
);

export default router;
