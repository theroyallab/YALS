import { loadModel, model } from "./common/modelContainer.ts";
import { ModelConfig } from "./common/configModels.ts";
import { BaseSamplerRequest } from "./common/sampling.ts";

// Create the model configuration matching the Model.init expectations
const modelConfig = ModelConfig.parse({
    model_dir: "/home/blackroot/Desktop/YALS/bindings/gguf/",
    model_name: "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    num_gpu_layers: 999,
    max_seq_len: undefined,
});

// Load the model with the new configuration
await loadModel(modelConfig);

const samplerRequest = BaseSamplerRequest.parse({
    temperature: 0,
    max_tokens: 200,
});

let abort = new AbortController();
const encoder = new TextEncoder();
let buffer = "";

const prompt =
    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a robot<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRespond litrally with "Hi nice to meet you"<|eot_id|><|start_header_id|>assistant<|end_header_id|>';
for (let i = 0; i < 4; i++) {
    console.log();
    console.log("NEXT");
    console.log();

    for await (
        const chunk of model!.generateGen(prompt, samplerRequest, abort)
    ) {
        await Deno.stdout.write(encoder.encode(chunk));
        buffer += chunk;
    }
}
