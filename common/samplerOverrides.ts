import * as YAML from "@std/yaml";
import * as z from "@/common/myZod.ts";
import { logger } from "@/common/logging.ts";

export const SamplerOverride = z.object({
    override: z.unknown().refine((val) => val !== undefined && val !== null, {
        message: "Override value cannot be undefined or null",
    }),
    force: z.boolean().optional().default(false),
    additive: z.boolean().optional().default(false),
});

// Sampler overrides
export type SamplerOverride = z.infer<typeof SamplerOverride>;

class SamplerOverridesContainer {
    selectedPreset?: string;
    overrides: Record<string, SamplerOverride> = {};
}

export const overridesContainer = new SamplerOverridesContainer();

export function overridesFromDict(newOverrides: Record<string, unknown>) {
    const parsedOverrides: Record<string, SamplerOverride> = {};

    // Validate each entry as a SamplerOverride type
    for (const [key, value] of Object.entries(newOverrides)) {
        try {
            parsedOverrides[key] = SamplerOverride.parse(value);
        } catch (error) {
            if (error instanceof Error) {
                logger.error(error.stack);
                logger.warn(
                    `Skipped override with key "${key}"` +
                        "due to the above error.",
                );
            }
        }
    }

    overridesContainer.overrides = parsedOverrides;
}

export async function overridesFromFile(presetName: string) {
    const presetPath = `sampler_overrides/${presetName}.yml`;
    overridesContainer.selectedPreset = presetName;

    // Read from override preset file
    const fileInfo = await Deno.stat(presetPath).catch(() => null);
    if (fileInfo?.isFile) {
        const rawPreset = await Deno.readTextFile(presetPath);
        const presetsYaml = YAML.parse(rawPreset) as Record<string, unknown>;

        overridesFromDict(presetsYaml);

        logger.info(`Applied sampler overrides from preset ${presetName}`);
    } else {
        throw new Error(
            `Sampler override file named ${presetName} was not found. ` +
                "Make sure it's located in the sampler_overrides folder.",
        );
    }
}

export function getSamplerDefault<T>(key: string, fallback?: T): T {
    const defaultValue = overridesContainer.overrides[key]?.override ??
        fallback;

    return defaultValue as T;
}

// Link resolver to Zod
z.registerSamplerOverrideResolver(getSamplerDefault);
