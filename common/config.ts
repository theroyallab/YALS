import * as YAML from "@std/yaml";

import {
    ConfigSchema,
    LoggingConfig,
    ModelConfig,
    NetworkConfig,
} from "./configModels.ts";
import { logger } from "@/common/logging.ts";

// Initialize with an empty config
export let config: ConfigSchema = ConfigSchema.parse({
    network: NetworkConfig.parse({}),
    logging: LoggingConfig.parse({}),
    model: ModelConfig.parse({}),
});

export async function loadConfig(args: Record<string, unknown>) {
    const configPath = "config.yml";

    const fileInfo = await Deno.stat(configPath).catch(() => null);
    if (!fileInfo?.isFile) {
        logger.warn("Could not find a config file. Starting anyway.");
        return;
    }

    const rawConfig = await Deno.readFile(configPath);
    const rawConfigStr = new TextDecoder().decode(rawConfig);

    const parsedConfig = YAML.parse(rawConfigStr) as Record<string, unknown>;
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
