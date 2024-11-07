import { Model } from "../bindings/bindings.ts";

export let model: Model | undefined = undefined;
export async function loadModel(modelPath: string, gpuLayers: number) {
    model = await Model.init(modelPath, gpuLayers);
}
