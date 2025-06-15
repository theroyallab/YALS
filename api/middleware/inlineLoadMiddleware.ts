import { HonoRequest, Next } from "hono";
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
        newModelName && model?.path.name !== newModelName.replace(".gguf", "")
    ) {
        console.log("Loading inline");

        const modelLoadRequest = await ModelLoadRequest.parseAsync({
            model_name: newModelName,
        });

        await apiLoadModel(modelLoadRequest, req.raw.signal);
    }

    await next();
};

export default inlineLoadMiddleware;
