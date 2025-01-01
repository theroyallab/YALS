import { Model } from "@/bindings/bindings.ts";
import { ModelConfig } from "@/common/configModels.ts";
import { logger } from "@/common/logging.ts";

export let model: Model | undefined = undefined;

export async function loadModel(
    params: ModelConfig,
    progressCallback?: (progress: number) => boolean,
) {
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
