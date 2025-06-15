import { HonoRequest, Next } from "hono";
import { HTTPException } from "hono/http-exception";
import { getAuthPermission } from "@/common/auth.ts";
import { config } from "@/common/config.ts";
import { logger } from "@/common/logging.ts";
import { model } from "@/common/modelContainer.ts";

import { ModelLoadRequest } from "../core/types/model.ts";
import { apiLoadModel } from "../core/utils/model.ts";

// This middleware is only run after sValidator
// Not a real middleware since hono can't tell that ctx has validated params
// See: https://github.com/orgs/honojs/discussions/1050
const inlineLoadMiddleware = async (
    req: HonoRequest,
    next: Next,
    newModelName?: string,
) => {
    if (
        !newModelName || model?.path.name === newModelName.replace(".gguf", "")
    ) {
        await next();
        return;
    }

    const permission = getAuthPermission(req.header());

    // Check if inline loading is enabled
    if (!config.model.inline_model_loading) {
        if (permission === "admin") {
            logger.warn(
                `Unable to switch model to ${newModelName} because ` +
                    '"inline_model_loading" is not True in config.yml.',
            );
        }

        await next();
        return;
    }

    // Only allow admins to swap models
    if (permission !== "api") {
        throw new HTTPException(401, {
            message: `Unable to switch model to ${newModelName} ` +
                "because an admin key isn't provided",
        });
    }

    // Create a load request payload
    const modelLoadRequest = await ModelLoadRequest.parseAsync({
        model_name: newModelName,
    });

    await apiLoadModel(modelLoadRequest, req.raw.signal);

    await next();
};

export default inlineLoadMiddleware;
