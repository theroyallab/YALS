import * as YAML from "@std/yaml";

import { ConfigSchema } from "./configModels.ts";

// Initialize with an empty config
export let config: ConfigSchema = ConfigSchema.parse({});

export async function loadConfig() {
    const configPath = "config.yml";

    if (!(await Deno.stat(configPath))) {
        return;
    }

    const rawConfig = await Deno.readFile(configPath);
    const rawConfigStr = new TextDecoder().decode(rawConfig);

    const parsedConfig = YAML.parse(rawConfigStr);
    config = await ConfigSchema.parseAsync(parsedConfig);
}
