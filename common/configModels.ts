import { z } from "zod";
import "@/common/extendZod.ts";

export const NetworkConfig = z.object({
    host: z.string().nullish().coalesce("127.0.0.1"),
    port: z.number().nullish().coalesce(5000),
});

export type NetworkConfig = z.infer<typeof NetworkConfig>;

export const ModelConfig = z.object({
    model_dir: z.string().nullish().coalesce("models"),
    model_name: z.string().nullish(),
    num_gpu_layers: z.number().nullish().coalesce(0),
    max_seq_len: z.number().nullish(),
});

export type ModelConfig = z.infer<typeof ModelConfig>;

export const ConfigSchema = z.object({
    network: NetworkConfig,
    model: ModelConfig,
});

export type ConfigSchema = z.infer<typeof ConfigSchema>;

export const testSchema = z.object({
    firstValue: z.string().nullish().coalesce("Models"),
});
