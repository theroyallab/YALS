import { Hono } from "hono";
import { describeRoute } from "hono-openapi";
import { validator as zValidator } from "hono-openapi/zod";
import {
    ModelCard,
    ModelList,
    ModelLoadRequest,
} from "@/api/core/types/model.ts";
import { ModelConfig } from "@/common/configModels.ts";
import { config } from "@/common/config.ts";
import { logger } from "@/common/logging.ts";
import * as modelContainer from "@/common/modelContainer.ts";
import { jsonContent } from "@/common/networking.ts";

import checkModelMiddleware from "../middleware/checkModelMiddleware.ts";
import { z } from "zod";

const router = new Hono();

const modelsRoute = describeRoute({
    responses: {
        200: jsonContent(ModelList, "List of models in directory"),
    },
});

router.get(
    "/v1/models",
    modelsRoute,
    async (c) => {
        const modelCards: ModelCard[] = [];
        for await (const file of Deno.readDir(config.model.model_dir)) {
            if (!file.name.endsWith(".gguf")) {
                continue;
            }

            const modelCard = await ModelCard.parseAsync({
                id: file.name,
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

        // Load the model
        await modelContainer.loadModel(loadParams, progressCallback);
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
    checkModelMiddleware,
    async (c) => {
        await modelContainer.unloadModel();

        c.status(200);
        return c.body(null);
    },
);

export default router;
