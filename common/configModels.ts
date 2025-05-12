import * as z from "@/common/myZod.ts";
import { GGMLType } from "@/bindings/types.ts";

export const NetworkConfig = z.object({
    host: z.string().nullish().coalesce("127.0.0.1"),
    port: z.number().nullish().coalesce(5000),
    disable_auth: z.boolean().nullish().coalesce(false),
});

export type NetworkConfig = z.infer<typeof NetworkConfig>;

export const LoggingConfig = z.object({
    log_prompt: z.boolean().nullish().coalesce(false),
    log_generation_params: z.boolean().nullish().coalesce(false),
});

export type LoggingConfig = z.infer<typeof LoggingConfig>;

export const ModelConfig = z.object({
    model_dir: z.string().nullish().coalesce("models"),
    model_name: z.string().nullish(),
    use_as_default: z.array(z.string()).nullish().coalesce([]),
    max_seq_len: z.number().nullish(),
    num_slots: z.number().nullish().coalesce(1),
    cache_size: z.number().nullish(),
    chunk_size: z.number().nullish().coalesce(512),
    num_gpu_layers: z.number().nullish().coalesce(0),
    gpu_split: z.array(z.number()).nullish().coalesce([]),
    num_threads: z.number().nullish().coalesce(-1),
    prompt_template: z.string().nullish(),
    flash_attention: z.boolean().nullish().coalesce(false),
    rope_freq_base: z.number().nullish().coalesce(0),
    enable_yarn: z.boolean().nullish().coalesce(false),
    cache_mode_k: z.union([
        z.string().transform((str) =>
            GGMLType[str.toLowerCase() as keyof typeof GGMLType]
        ),
        z.number(),
    ])
        .nullish().coalesce(GGMLType.f16),
    cache_mode_v: z.union([
        z.string().transform((str) =>
            GGMLType[str.toLowerCase() as keyof typeof GGMLType]
        ),
        z.number(),
    ])
        .nullish().coalesce(GGMLType.f16),
    override_tensor: z.string().nullish(),
    mmap: z.boolean().nullish().coalesce(true),
});

export type ModelConfig = z.infer<typeof ModelConfig>;

export const SamplingConfig = z.object({
    override_preset: z.string().nullish(),
});

export type SamplingConfig = z.infer<typeof SamplingConfig>;

export const ConfigSchema = z.object({
    network: NetworkConfig,
    logging: LoggingConfig,
    model: ModelConfig,
    sampling: SamplingConfig,
});

export type ConfigSchema = z.infer<typeof ConfigSchema>;
