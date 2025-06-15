import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { describeRoute } from "hono-openapi";
import { validator as sValidator } from "hono-openapi";
import { AuthPermissionResponse } from "@/api/core/types/auth.ts";
import { HealthSchema } from "@/api/core/types/health.ts";
import { applyChatTemplate } from "@/api/OAI/utils/chatCompletion.ts";
import {
    ModelCard,
    ModelList,
    ModelLoadRequest,
} from "@/api/core/types/model.ts";
import {
    TemplateList,
    TemplateSwitchRequest,
} from "@/api/core/types/template.ts";
import {
    TokenDecodeRequest,
    TokenDecodeResponse,
    TokenEncodeRequest,
    TokenEncodeResponse,
} from "@/api/core/types/token.ts";
import { AuthKeyPermission, getAuthPermission } from "@/common/auth.ts";
import { config } from "@/common/config.ts";
import * as modelContainer from "@/common/modelContainer.ts";
import { jsonContent, toHttpException } from "@/common/networking.ts";
import { PromptTemplate } from "@/common/templating.ts";

import authMiddleware from "../middleware/authMiddleware.ts";
import checkModelMiddleware from "../middleware/checkModelMiddleware.ts";
import { apiLoadModel } from "./utils/model.ts";

const router = new Hono();

const healthRoute = describeRoute({
    responses: {
        200: jsonContent(HealthSchema, "Health status of server"),
    },
});

router.get(
    "/health",
    healthRoute,
    checkModelMiddleware,
    (c) => {
        return c.json(HealthSchema.parse({ health: "ok" }));
    },
);

const modelsRoute = describeRoute({
    responses: {
        200: jsonContent(ModelList, "List of models in directory"),
    },
});

router.on(
    "GET",
    ["/v1/models", "/v1/model/list"],
    modelsRoute,
    authMiddleware(AuthKeyPermission.API),
    async (c) => {
        const modelCards: ModelCard[] = [];
        for await (const file of Deno.readDir(config.model.model_dir)) {
            if (!file.name.endsWith(".gguf")) {
                continue;
            }

            const modelCard = ModelCard.parse({
                id: file.name.replace(".gguf", ""),
            });

            modelCards.push(modelCard);
        }

        const modelList = ModelList.parse({
            data: modelCards,
        });

        return c.json(modelList);
    },
);

const currentModelRoute = describeRoute({
    responses: {
        200: jsonContent(
            ModelCard,
            "The currently loaded model (if it exists)",
        ),
    },
});

router.get(
    "/v1/model",
    currentModelRoute,
    authMiddleware(AuthKeyPermission.API),
    checkModelMiddleware,
    (c) => {
        const modelCard = ModelCard.parse({
            id: c.var.model.path.base,
        });

        return c.json(modelCard);
    },
);

const loadModelRoute = describeRoute({
    responses: {
        200: {
            description: "Model successfully loaded",
        },
    },
});

router.post(
    "/v1/model/load",
    loadModelRoute,
    authMiddleware(AuthKeyPermission.Admin),
    sValidator("json", ModelLoadRequest),
    async (c) => {
        const params = c.req.valid("json");

        // Pass to common function
        await apiLoadModel(params, c.req.raw.signal);

        c.status(200);
        return c.body(null);
    },
);

const unloadRoute = describeRoute({
    responses: {
        200: {
            description: "Model successfully unloaded",
        },
    },
});

router.post(
    "/v1/model/unload",
    unloadRoute,
    authMiddleware(AuthKeyPermission.Admin),
    checkModelMiddleware,
    async (c) => {
        await modelContainer.unloadModel(true);

        c.status(200);
        return c.body(null);
    },
);

const templatesRoute = describeRoute({
    responses: {
        200: jsonContent(TemplateList, "List of prompt templates"),
    },
});

router.on(
    "GET",
    ["/v1/templates", "/v1/template/list"],
    templatesRoute,
    authMiddleware(AuthKeyPermission.API),
    async (c) => {
        const templates: string[] = [];
        for await (const file of Deno.readDir("templates")) {
            if (!file.name.endsWith(".jinja")) {
                continue;
            }

            templates.push(file.name.replace(".jinja", ""));
        }

        const templateList = TemplateList.parse({
            data: templates,
        });

        return c.json(templateList);
    },
);

const templateSwitchRoute = describeRoute({
    responses: {
        200: {
            description: "Prompt template switched",
        },
    },
});

router.post(
    "/v1/template/switch",
    templateSwitchRoute,
    authMiddleware(AuthKeyPermission.API),
    checkModelMiddleware,
    sValidator("json", TemplateSwitchRequest),
    async (c) => {
        const params = c.req.valid("json");

        const templatePath = `templates/${params.prompt_template_name}`;
        c.var.model.promptTemplate = await PromptTemplate.fromFile(
            templatePath,
        );
    },
);

const authPermissionRoute = describeRoute({
    responses: {
        200: jsonContent(
            AuthPermissionResponse,
            "Returns permissions of a given auth key",
        ),
    },
});

router.get(
    "/v1/auth/permission",
    authPermissionRoute,
    authMiddleware(AuthKeyPermission.API),
    (c) => {
        try {
            const permission = getAuthPermission(c.req.header());
            const response = AuthPermissionResponse.parse({
                permission,
            });

            return c.json(response);
        } catch (error) {
            throw toHttpException(error, 400);
        }
    },
);

const tokenEncodeRoute = describeRoute({
    responses: {
        200: jsonContent(TokenEncodeResponse, "Encode token response"),
    },
});

router.post(
    "/v1/token/encode",
    tokenEncodeRoute,
    authMiddleware(AuthKeyPermission.API),
    checkModelMiddleware,
    sValidator("json", TokenEncodeRequest),
    async (c) => {
        const params = c.req.valid("json");

        let text: string;
        if (typeof params.text === "string") {
            text = params.text;
        } else if (Array.isArray(params.text)) {
            if (!c.var.model.promptTemplate) {
                throw new HTTPException(422, {
                    message: "Cannot tokenize chat completion " +
                        "because a prompt template is not set",
                });
            }

            text = applyChatTemplate(
                c.var.model,
                c.var.model.promptTemplate,
                params.text,
                {
                    addBosToken: params.add_bos_token,
                    addGenerationPrompt: false,
                },
            );
        } else {
            throw new HTTPException(422, {
                message: "Unable to tokenize the provided text. " +
                    "Check your formatting?",
            });
        }

        const tokens = await c.var.model.tokenizer.tokenize(
            text,
            params.add_bos_token,
            params.encode_special_tokens,
        );

        const resp = TokenEncodeResponse.parse({
            tokens,
            length: tokens.length,
        });

        return c.json(resp);
    },
);

const tokenDecodeRoute = describeRoute({
    responses: {
        200: jsonContent(TokenDecodeResponse, "Decode token response"),
    },
});

router.post(
    "/v1/token/decode",
    tokenDecodeRoute,
    authMiddleware(AuthKeyPermission.API),
    checkModelMiddleware,
    sValidator("json", TokenDecodeRequest),
    async (c) => {
        const params = c.req.valid("json");

        const text = await c.var.model.tokenizer.detokenize(
            params.tokens,
            undefined,
            params.add_bos_token,
            params.decode_special_tokens,
        );

        const resp = TokenDecodeResponse.parse({
            text,
        });

        return c.json(resp);
    },
);

export default router;
