import { Model } from "@/bindings/bindings.ts";
import { ModelConfig } from "@/common/configModels.ts";

export let model: Model | undefined = undefined;

export async function loadModel(params: ModelConfig) {
    model = await Model.init(params);
}

export async function unloadModel() {
    await model?.unload();
    model = undefined;
}
