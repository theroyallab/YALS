import { config } from "@/common/config.ts";
import { ModelConfig } from "@/common/configModels.ts";
import { logger } from "@/common/logging.ts";
import { ModelLoadRequest } from "../types/model.ts";
import * as modelContainer from "@/common/modelContainer.ts";
import { HTTPException } from "hono/http-exception";

export async function apiLoadModel(
    params: ModelLoadRequest,
    requestSignal: AbortSignal,
) {
    const combinedParams = ModelConfig.parse({
        ...params,
        model_dir: config.model.model_dir,
    });

    // Makes sure the event doesn't fire multiple times
    let finished = false;

    // Abort handler
    const progressAbort = new AbortController();
    requestSignal.addEventListener("abort", () => {
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
        await modelContainer.loadModel(combinedParams, progressCallback);
    } catch (error) {
        if (error instanceof Error) {
            throw new HTTPException(422, error);
        }
    }

    finished = true;
}
