import { loadModel, model } from "./common/modelContainer.ts";
import { BaseSamplerRequest } from "./common/sampling.ts";

await loadModel("/home/blackroot/Desktop/YALS/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", 999);

const samplerRequest = BaseSamplerRequest.parse({
    temperature: 1.2,
    top_k: 40,
    top_p: 1,
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