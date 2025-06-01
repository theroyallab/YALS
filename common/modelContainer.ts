import { Mutex } from "@core/asyncutil";
import * as Path from "@std/path";
import * as YAML from "@std/yaml";

import * as z from "./myZod.ts";
import { Model } from "@/bindings/bindings.ts";
import { config } from "./config.ts";
import { InlineConfigSchema, ModelConfig } from "./configModels.ts";
import { logger } from "./logging.ts";

export let model: Model | undefined = undefined;
const loadLock = new Mutex();

export async function loadModel(
    params: ModelConfig,
    progressCallback?: (progress: number) => boolean,
) {
    if (loadLock.locked) {
        throw new Error(
            "Another model load operation is in progress. Please wait.",
        );
    }

    using _lock = await loadLock.acquire();

    if (model) {
        if (model?.path.name === params.model_name?.replace(".gguf", "")) {
            throw new Error(
                `Model ${params.model_name} is already loaded! Aborting.`,
            );
        }

        logger.info("Unloading existing model.");
        await unloadModel();
    }

    if (!params.model_name?.endsWith(".gguf")) {
        params.model_name = `${params.model_name}.gguf`;
    }

    model = await Model.init(params, progressCallback);
}

export async function unloadModel(skipQueue: boolean = false) {
    await model?.unload(skipQueue);
    model = undefined;
}

// Applies model load overrides. Sources are inline and model config
// Agnostic due to passing of ModelLoadRequest and ModelConfig
export async function applyLoadDefaults(item: unknown) {
    const obj = z.record(z.string(), z.unknown()).safeParse(item);

    // Silently return since further validation will fail
    if (!obj.success) {
        return item;
    }

    const data = { ...obj.data };
    const modelOverrides: Record<string, unknown> = {};

    if (typeof data["model_name"] === "string") {
        const modelName = data["model_name"] as string;
        const modelDir = data["model_dir"] as string ?? config.model.model_dir;
        const inlineConfigPath = Path.join(
            modelDir,
            `${modelName.replace(".gguf", "")}.yml`,
        );

        const fileInfo = await Deno.stat(inlineConfigPath).catch(() => null);
        if (fileInfo?.isFile) {
            const rawInlineConfig = await Deno.readTextFile(inlineConfigPath);
            const inlineYaml = YAML.parse(rawInlineConfig) as Record<
                string,
                unknown
            >;

            const inlineResult = InlineConfigSchema.safeParse(inlineYaml);
            if (inlineResult.success) {
                Object.assign(modelOverrides, inlineResult.data.model);
            } else {
                logger.warn(
                    `Invalid inline config for ${modelName}: ` +
                        inlineResult.error.message,
                );
            }
        }
    }

    // Iterate through defaults
    for (const key of config.model.use_as_default) {
        if (key in config.model) {
            modelOverrides[key] =
                config.model[key as keyof typeof config.model];
        }
    }

    // Apply modelOverrides first then overlay data
    return { ...modelOverrides, ...data };
}
