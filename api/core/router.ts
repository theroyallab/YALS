import { Hono } from "hono";
import { describeRoute } from "hono-openapi";
import { validator as zValidator } from "hono-openapi/zod";
import { ModelLoadRequest } from "@/api/core/types/model.ts";
import { ModelConfig } from "@/common/configModels.ts";
import { config } from "@/common/config.ts";
import { logger } from "@/common/logging.ts";
import * as modelContainer from "@/common/modelContainer.ts";

import checkModelMiddleware from "../middleware/checkModelMiddleware.ts";

const router = new Hono();

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
