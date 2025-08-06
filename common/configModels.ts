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
    log_requests: z.boolean().nullish().coalesce(false),
});

export type LoggingConfig = z.infer<typeof LoggingConfig>;

export const ModelConfig = z.object({
    model_dir: z.string().nullish().coalesce("models"),
    inline_model_loading: z.boolean().nullish().coalesce(false),
    use_dummy_models: z.boolean().nullish().coalesce(false),
    dummy_model_names: z.array(z.string()).nullish().coalesce([
        "gpt-3.5-turbo",
    ]),
    model_name: z.string().cleanOptional(),
    use_as_default: z.array(z.string()).nullish().coalesce([]),
    max_seq_len: z.number().nullish().coalesce(4096),
    num_slots: z.number().nullish().coalesce(1),
    cache_size: z.number().cleanOptional(),
    chunk_size: z.number().nullish().coalesce(512),
    num_gpu_layers: z.number().nullish().coalesce(0),
    gpu_split: z.array(z.number()).nullish().coalesce([]),
    tensor_parallel: z.boolean().nullish().coalesce(false),
    num_threads: z.number().nullish().coalesce(-1),
    prompt_template: z.string().cleanOptional(),
    flash_attention: z.boolean().nullish().coalesce(true),
    rope_freq_base: z.number().nullish().coalesce(0),
    enable_yarn: z.boolean().nullish().coalesce(false),
    cache_mode_k: z.union([
        z.string().transform((str) =>
            GGMLType[str.toLowerCase() as keyof typeof GGMLType]
        ),
        z.number(),
    ])
        .nullish()
        .coalesce(GGMLType.f16),
    cache_mode_v: z.union([
        z.string().transform((str) =>
            GGMLType[str.toLowerCase() as keyof typeof GGMLType]
        ),
        z.number(),
    ])
        .nullish()
        .coalesce(GGMLType.f16),
    kv_offload: z.boolean().nullish().coalesce(true),
    override_tensor: z.array(z.string()).nullish().coalesce([]),
    n_cpu_moe: z.union([z.number(), z.literal("all")]).cleanOptional(),
    mmap: z.boolean().nullish().coalesce(true),
});

export type ModelConfig = z.infer<typeof ModelConfig>;

// TODO: Maybe remove the extend and add to ModelLoadRequest
export const StrippedModelConfig = ModelConfig.omit({
    model_dir: true,
    inline_model_loading: true,
    use_dummy_models: true,
    dummy_model_names: true,
    use_as_default: true,
});

export type StrippedModelConfig = z.infer<typeof StrippedModelConfig>;

export const SamplingConfig = z.object({
    override_preset: z.string().cleanOptional(),
});

export type SamplingConfig = z.infer<typeof SamplingConfig>;

export const DeveloperConfig = z.object({
    realtime_process_priority: z.boolean().nullish().coalesce(true),
});

export const ConfigSchema = z.object({
    network: NetworkConfig,
    logging: LoggingConfig,
    model: ModelConfig,
    sampling: SamplingConfig,
    developer: DeveloperConfig,
});

export type ConfigSchema = z.infer<typeof ConfigSchema>;

// Config shim for inline overrides
export const InlineConfigSchema = z.object({
    model: z.record(z.string(), z.unknown()),
});

export type InlineConfigSchema = z.infer<typeof InlineConfigSchema>;
