import { Mutex } from "@core/asyncutil";

import { Model } from "@/bindings/bindings.ts";
import { ModelConfig } from "@/common/configModels.ts";
import { logger } from "@/common/logging.ts";

export let model: Model | undefined = undefined;
const loadLock = new Mutex();

export async function loadModel(
    params: ModelConfig,
    progressCallback?: (progress: number) => boolean,
) {
    if (loadLock.locked) {
        throw new Error("Another model load operation is in progress. Please wait.");
    }

    using _lock = await loadLock.acquire()

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
