import { Mutex } from "@core/asyncutil";

import * as z from "./myZod.ts";
import { Model } from "@/bindings/bindings.ts";
import { config } from "./config.ts";
import { ModelConfig } from "./configModels.ts";
import { logger } from "./logging.ts";

export let model: Model | undefined = undefined;
const loadLock = new Mutex();

export async function loadModel(
    params: ModelConfig,
    progressCallback?: (progress: number) => boolean,
) {
    if (loadLock.locked) {
        throw new Error(
            "Another model load operation is in progress. Please wait.",
        );
    }

    using _lock = await loadLock.acquire();

    if (model) {
        if (model?.path.name === params.model_name?.replace(".gguf", "")) {
            throw new Error(
                `Model ${params.model_name} is already loaded! Aborting.`,
            );
        }

        logger.info("Unloading existing model.");
        await unloadModel();
    }

    if (!params.model_name?.endsWith(".gguf")) {
        params.model_name = `${params.model_name}.gguf`;
    }

    model = await Model.init(params, progressCallback);
}

export async function unloadModel(skipQueue: boolean = false) {
    await model?.unload(skipQueue);
    model = undefined;
}

// Applies model load overrides. Sources are inline and model config
// Agnostic due to passing of ModelLoadRequest and ModelConfig
// TODO: Add inline loading defaults
export function applyLoadDefaults(item: unknown) {
    const obj = z.record(z.string(), z.unknown()).safeParse(item);
    if (obj.success) {
        const data = { ...obj.data };

        // Iterate through use_as_default
        // TODO: Move config.model assert into ModelConfig with Zod 4
        for (const key of config.model.use_as_default) {
            if (
                (data[key] === undefined || data[key] === null) &&
                key in config.model
            ) {
                data[key] = config.model[key as keyof typeof config.model];
            }
        }

        return data;
    }

    // Silently return item if obj parse fails
    return item;
}
