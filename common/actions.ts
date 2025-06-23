import { config } from "./config.ts";
import * as YAML from "@std/yaml";

import { InlineConfigSchema, StrippedModelConfig } from "./configModels.ts";

export async function createInlineConfig() {
    if (!config.model.model_name) {
        throw new Error(
            "Please provide a model name to generate an inline config.",
        );
    }

    // Check if an inline config already exists and exit if so
    const inlineConfigPath = config.model.model_dir +
        `${config.model.model_name}.yml`;
    const fileInfo = await Deno.stat(inlineConfigPath).catch(() => null);
    if (!fileInfo?.isFile) {
        throw new Error(
            "Inline config already found for provided model." +
                " Delete it and try again",
        );
    }

    const strippedModelConfig = StrippedModelConfig.omit({
        model_name: true,
    }).parse(config.model);

    const inlineConfig: InlineConfigSchema = { model: strippedModelConfig };

    const inlineYaml = YAML.stringify(inlineConfig);
    await Deno.writeTextFile(inlineConfigPath, inlineYaml);
}
