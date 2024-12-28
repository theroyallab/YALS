import * as YAML from "@std/yaml";
import { parse, parseAsync } from "valibot";

import { ConfigSchema, ModelConfig, NetworkConfig } from "./configModels.ts";

// Initialize with an empty config
export let config: ConfigSchema = parse(ConfigSchema, {
    network: parse(NetworkConfig, {}),
    model: parse(ModelConfig, {}),
});

export async function loadConfig() {
    const configPath = "config.yml";

    if (!(await Deno.stat(configPath))) {
        return;
    }

    const rawConfig = await Deno.readFile(configPath);
    const rawConfigStr = new TextDecoder().decode(rawConfig);

    const parsedConfig = YAML.parse(rawConfigStr);
    config = await parseAsync(ConfigSchema, parsedConfig);
}
