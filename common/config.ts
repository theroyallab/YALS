import * as YAML from "@std/yaml";

import { ConfigSchema, ModelConfig, NetworkConfig } from "./configModels.ts";
import { logger } from "@/common/logging.ts";

// Initialize with an empty config
export let config: ConfigSchema = ConfigSchema.parse({
    network: NetworkConfig.parse({}),
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

    // Merge args with config
    for (const groupName of Object.keys(parsedConfig)) {
        const group = parsedConfig[groupName] as Record<string, unknown>;

        if (args[groupName]) {
            parsedConfig[groupName] = { ...group, ...args[groupName] }
        }
    }

    // Parse config
    config = await ConfigSchema.parseAsync(parsedConfig);
}
