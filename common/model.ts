import { ModelContainer } from "../bindings/bindings.ts";

export let model: ModelContainer | undefined = undefined;

export async function loadModel(modelPath: string, gpuLayers: 999) {
    model = await ModelContainer.init(modelPath, gpuLayers);
}
