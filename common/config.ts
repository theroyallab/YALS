import * as YAML from "@std/yaml";

import {
    ConfigSchema,
    LoggingConfig,
    ModelConfig,
    NetworkConfig,
    SamplingConfig,
} from "./configModels.ts";
import { logger } from "@/common/logging.ts";

// Initialize with an empty config
export let config: ConfigSchema = ConfigSchema.parse({
    network: NetworkConfig.parse({}),
    logging: LoggingConfig.parse({}),
    model: ModelConfig.parse({}),
    sampling: SamplingConfig.parse({}),
});

export async function loadConfig(args: Record<string, unknown>) {
    const configPath = "config.yml";
    let parsedConfig: Record<string, unknown> = {};

    // Warn if the file doesn't exist
    const fileInfo = await Deno.stat(configPath).catch(() => null);
    if (fileInfo?.isFile) {
        const rawConfig = await Deno.readTextFile(configPath);
        parsedConfig = YAML.parse(rawConfig) as Record<string, unknown>;
    } else {
        logger.warn("Could not find a config file. Starting anyway.");
    }

    const mergedConfig: Record<string, unknown> = {};

    // Single loop to merge default config, file config, and args
    for (const key of Object.keys(config) as Array<keyof typeof config>) {
        mergedConfig[key] = {
            ...config[key],
            ...(parsedConfig[key] as Record<string, unknown> || {}),
            ...(args[key] as Record<string, unknown> || {}),
        };
    }

    // Parse merged config
    config = ConfigSchema.parse(mergedConfig);
}
