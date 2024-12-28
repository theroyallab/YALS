import * as v from "valibot";

export const NetworkConfig = v.object({
    host: v.nullish(v.string(), "127.0.0.1"),
    port: v.nullish(v.number(), 5000),
});

export type NetworkConfig = v.InferOutput<typeof NetworkConfig>;

export const ModelConfig = v.object({
    model_dir: v.nullish(v.string(), "models"),
    model_name: v.nullish(v.string()),
    num_gpu_layers: v.nullish(v.number(), 0),
    max_seq_len: v.nullish(v.number()),
});

export type ModelConfig = v.InferOutput<typeof ModelConfig>;

export const ConfigSchema = v.object({
    network: NetworkConfig,
    model: ModelConfig,
});

export type ConfigSchema = v.InferOutput<typeof ConfigSchema>;
