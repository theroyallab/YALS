import commandLineArgs from "command-line-args";
import * as Path from "@std/path";
import * as YAML from "@std/yaml";
import { config } from "./config.ts";
import { InlineConfigSchema, StrippedModelConfig } from "./configModels.ts";
import { logger } from "./logging.ts";

export async function genInlineConfigAction() {
    if (!config.model.model_name) {
        throw new Error(
            "Please provide a model name to generate an inline config.",
        );
    }

    const modelName = config.model.model_name.replace(".gguf", "");

    // Check if an inline config already exists and exit if so
    const inlineConfigPath = Path.join(
        config.model.model_dir,
        `${modelName}.yml`,
    );

    const fileInfo = await Deno.stat(inlineConfigPath).catch(() => null);
    if (fileInfo?.isFile) {
        throw new Error(
            "Inline config already found for " +
                `provided model name. ${modelName}` +
                "Delete it and try again.",
        );
    }

    const strippedModelConfig = StrippedModelConfig.omit({
        model_name: true,
    }).parse(config.model);

    const inlineConfig: InlineConfigSchema = { model: strippedModelConfig };

    const inlineYaml = YAML.stringify(inlineConfig);
    await Deno.writeTextFile(inlineConfigPath, inlineYaml);

    logger.info(`Successfully wrote inline config to ${inlineConfigPath}`);
}

export async function runAction(args: commandLineArgs.CommandLineOptions) {
    switch (args.actions.subcommand) {
        case "gen-inline-config":
            await genInlineConfigAction();
            return true;
        default:
            return false;
    }
}
