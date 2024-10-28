import { parse as parseYaml } from "@std/yaml";
import { lib, ReadbackBuffer, SamplerBuilder } from "./bindings/bindings.ts";

// Learn more at https://docs.deno.com/runtime/manual/examples/module_metadata#concepts
if (import.meta.main) {
    // Read YAML config
    let configFile: string;

    // Surely there's a better way to do this in TS
    try {
        configFile = await Deno.readTextFile("./config.yaml");
    } catch {
        try {
            configFile = await Deno.readTextFile("./config.yml");
        } catch {
            throw new Error("No YAML config file found.");
        }
    }
    const config = parseYaml(configFile) as {
        modelPath: string;
        numberGpuLayers: number;
        contextLength: number;
        numBatches: number;
    };

    const { modelPath, numberGpuLayers, contextLength, numBatches } = config;
    const modelPathPtr = new TextEncoder().encode(modelPath + "\0");

    const llamaModel = await lib.symbols.LoadModel(
        Deno.UnsafePointer.of(modelPathPtr),
        numberGpuLayers,
    );

    const context = await lib.symbols.InitiateCtx(
        llamaModel,
        contextLength,
        numBatches,
    );

    // GreedySampler
    const samplerBuilder = new SamplerBuilder(lib, llamaModel);
    const sampler = samplerBuilder
        .tempSampler(1.0)
        .topK(40)
        .distSampler(1337)
        .build();

    const prompt = "Once upon a time";
    const promptPtr = new TextEncoder().encode(prompt + "\0");

    const readbackBuffer = new ReadbackBuffer(lib);

    // Don't await due to async generator
    lib.symbols.InferToReadbackBuffer(
        llamaModel,
        sampler,
        context,
        readbackBuffer.bufferPtr,
        Deno.UnsafePointer.of(promptPtr),
        150,
    );

    // Read from the read buffer
    for await (const nextString of readbackBuffer.read()) {
        const encoder = new TextEncoder();
        Deno.stdout.writeSync(encoder.encode(nextString));
    }

    // Close the library when done
    lib.close();
}
