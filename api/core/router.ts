import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { describeRoute } from "hono-openapi";
import { validator as zValidator } from "hono-openapi/zod";
import { AuthPermissionResponse } from "@/api/core/types/auth.ts";
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
import { ModelConfig } from "@/common/configModels.ts";
import { config } from "@/common/config.ts";
import { logger } from "@/common/logging.ts";
import * as modelContainer from "@/common/modelContainer.ts";
import { jsonContent } from "@/common/networking.ts";
import { PromptTemplate } from "@/common/templating.ts";

import authMiddleware from "../middleware/authMiddleware.ts";
import checkModelMiddleware from "../middleware/checkModelMiddleware.ts";

const router = new Hono();

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

            const modelCard = await ModelCard.parseAsync({
                id: file.name.replace(".gguf", ""),
            });

            modelCards.push(modelCard);
        }

        const modelList = await ModelList.parseAsync({
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
    async (c) => {
        const modelCard = await ModelCard.parseAsync({
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

// TODO: Make this a streaming response if necessary
router.post(
    "/v1/model/load",
    loadModelRoute,
    authMiddleware(AuthKeyPermission.Admin),
    zValidator("json", ModelLoadRequest),
    async (c) => {
        const params = c.req.valid("json");
        const loadParams = await ModelConfig.parseAsync({
            ...params,
            model_dir: config.model.model_dir,
        });

        // Makes sure the event doesn't fire multiple times
        let finished = false;

        // Abort handler
        const progressAbort = new AbortController();
        c.req.raw.signal.addEventListener("abort", () => {
            if (!finished) {
                progressAbort.abort();
            }
        });

        const progressCallback = (_progress: number): boolean => {
            if (progressAbort.signal.aborted) {
                logger.error("Load request cancelled");
                return false;
            }

            return true;
        };

        // Load the model and re-raise errors
        try {
            await modelContainer.loadModel(loadParams, progressCallback);
        } catch (error) {
            if (error instanceof Error) {
                throw new HTTPException(422, error);
            }
        }

        finished = true;

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

        const templateList = await TemplateList.parseAsync({
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
    zValidator("json", TemplateSwitchRequest),
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
    async (c) => {
        try {
            const permission = getAuthPermission(c.req.header());
            const response = await AuthPermissionResponse.parseAsync({
                permission,
            });

            return c.json(response);
        } catch (error) {
            if (error instanceof Error) {
                throw new HTTPException(400, error);
            }
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
    zValidator("json", TokenEncodeRequest),
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

        const tokens = await c.var.model.tokenize(
            text,
            params.add_bos_token,
            params.encode_special_tokens,
        );

        const resp = await TokenEncodeResponse.parseAsync({
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
    zValidator("json", TokenDecodeRequest),
    async (c) => {
        const params = c.req.valid("json");

        const text = await c.var.model.detokenize(
            params.tokens,
            undefined,
            params.add_bos_token,
            params.decode_special_tokens,
        );

        const resp = await TokenDecodeResponse.parseAsync({
            text,
        });

        return c.json(resp);
    },
);

export default router;
