import { loadModel, model } from "./common/modelContainer.ts";
import { ModelConfig } from "./common/configModels.ts";
import { BaseSamplerRequest } from "./common/sampling.ts";

const fullModelPath = "/home/blackroot/Desktop/YALS/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";

// Create the model configuration matching the Model.init expectations
const modelConfig = ModelConfig.parse({
  model_dir: fullModelPath,  // The full path to the .gguf file
  model_name: undefined,     // Not needed since we're using full path in model_dir
  num_gpu_layers: 999,
  max_seq_len: undefined
});

// Load the model with the new configuration
await loadModel(modelConfig);

const samplerRequest = BaseSamplerRequest.parse({
    temperature: 0,
    max_tokens: 35
});

const encoder = new TextEncoder();
let buffer = '';
for (let i = 0; i < 4; i++) {
  console.log();
  console.log("NEXT");
  console.log();
  
  for await (const chunk of model!.generateGen("Hi my name is", samplerRequest)) {
    await Deno.stdout.write(encoder.encode(chunk));
    buffer += chunk;
  }
}