import * as YAML from "@std/yaml";
import * as z from "@/common/myZod.ts";
import { logger } from "@/common/logging.ts";

// Sampling schemas
const GenerationOptionsSchema = z.aliasedObject(
    z.object({
        max_tokens: z.number().gte(0).nullish()
            .openapi({
                description: "Aliases: max_length",
            }),
        stop: z.union([
            z.string().transform((str) => [str]),
            z.array(z.union([z.string(), z.number()])),
        ])
            .nullish().coalesce([])
            .openapi({
                description: "Aliases: stop_sequence",
            }),
        add_bos_token: z.boolean().nullish().coalesce(true),
        ban_eos_token: z.boolean().nullish().coalesce(false)
            .openapi({
                description: "Aliases: ignore_eos",
            }),
        skip_special_tokens: z.boolean().nullish().coalesce(false),
        seed: z.number().nullish(),
        logit_bias: z.record(z.string(), z.number()).nullish().coalesce({}),
        grammar_string: z.string().nullish(),
        banned_tokens: z.union([
            z.array(z.number()),
            z.string()
                .transform((str) =>
                    str.replaceAll(" ", "")
                        .split(",")
                        .filter((x) => /^\d+$/.test(x))
                        .map((x) => parseInt(x))
                ),
        ])
            .nullish().coalesce([])
            .openapi({
                description: "Aliases: custom_token_bans",
            }),
        banned_strings: z.union([
            z.string().transform((str) => [str]),
            z.array(z.string()),
        ])
            .nullish().coalesce([]),
    }),
    [
        { field: "max_tokens", aliases: ["max_length"] },
        { field: "ban_eos_token", aliases: ["ignore_eos"] },
        { field: "stop", aliases: ["stop_sequence"] },
        { field: "banned_tokens", aliases: ["custom_token_bans"] },
    ],
)
    .openapi({
        description: "Generation options",
    });

const TemperatureSamplerSchema = z.object({
    temperature: z.number().gte(0).nullish().coalesce(1),
    temperature_last: z.boolean().nullish().coalesce(false),
})
    .openapi({
        description: "Temperature options",
    });

const AlphabetSamplerSchema = z.aliasedObject(
    z.object({
        top_k: z.number().gte(-1).transform((top_k) => top_k == -1 ? 0 : top_k)
            .nullish().coalesce(0),
        top_p: z.number().gte(0).lte(1).nullish().coalesce(1),
        min_p: z.number().gte(0).lte(1).nullish().coalesce(0),
        typical: z.number().gt(0).lte(1).nullish().coalesce(1),
        nsigma: z.number().gte(0).nullish().coalesce(0),
    }),
    [{ field: "typical", aliases: ["typical_p"] }],
)
    .openapi({
        description: "Alphabet samplers",
    });

const PenaltySamplerSchema = z.aliasedObject(
    z.object({
        frequency_penalty: z.number().gte(0).nullish().coalesce(0),
        presence_penalty: z.number().gte(0).nullish().coalesce(0),
        repetition_penalty: z.number().gt(0).nullish().coalesce(1)
            .openapi({
                description: "Aliases: rep_pen",
            }),
        penalty_range: z.number().nullish().coalesce(-1)
            .openapi({
                description:
                    "Aliases: repetition_range, repetition_penalty_range, rep_pen_range",
            }),
    }),
    [
        { field: "repetition_penalty", aliases: ["rep_pen"] },
        {
            field: "penalty_range",
            aliases: [
                "repetition_range",
                "repetition_penalty_range",
                "rep_pen_range",
            ],
        },
    ],
)
    .openapi({
        description: "Penalty samplers",
    });

const DrySchema = z.aliasedObject(
    z.object({
        dry_multiplier: z.number().nullish().coalesce(0),
        dry_base: z.number().nullish().coalesce(0),
        dry_allowed_length: z.number().nullish().coalesce(0),
        dry_sequence_breakers: z.union([
            z.string()
                .transform((str) => {
                    if (!str.startsWith("[")) {
                        str = `[${str}]`;
                    }

                    // Parse can fail, so return a default value if it does
                    try {
                        return JSON.parse(str);
                    } catch {
                        return [];
                    }
                }),
            z.array(z.string()),
        ])
            .nullish().coalesce([]),
        dry_range: z.number().nullish().coalesce(0)
            .openapi({
                description: "Aliases: dry_penalty_last_n",
            }),
    }),
    [{ field: "dry_range", aliases: ["dry_penalty_last_n"] }],
)
    .openapi({
        description: "DRY options",
    });

const XtcSchema = z.object({
    xtc_probability: z.number().nullish().coalesce(0),
    xtc_threshold: z.number().nullish().coalesce(0.1),
})
    .openapi({
        description: "XTC options",
    });

const DynatempSchema = z.aliasedObject(
    z.object({
        max_temp: z.number().gte(0).nullish().coalesce(1)
            .openapi({
                description: "Aliases: dynatemp_high",
            }),
        min_temp: z.number().gte(0).nullish().coalesce(1)
            .openapi({
                description: "Aliases: dynatemp_low",
            }),
        temp_exponent: z.number().gte(0).nullish().coalesce(1)
            .openapi({
                description: "Aliases: dynatemp_exponent",
            }),
    }),
    [
        { field: "max_temp", aliases: ["dynatemp_high"] },
        { field: "min_temp", aliases: ["dynatemp_low"] },
        { field: "temp_exponent", aliases: ["dynatemp_exponent"] },
    ],
)
    .openapi({
        description: "DynaTemp options",
    });

const MirostatSchema = z.object({
    mirostat_mode: z.number().nullish().coalesce(0),
    mirostat_tau: z.number().nullish().coalesce(1),
    mirostat_eta: z.number().nullish().coalesce(0),
})
    .openapi({
        description: "Mirostat options",
    });

// Construct from aliased sampler requests
export const BaseSamplerRequest = GenerationOptionsSchema
    .and(TemperatureSamplerSchema)
    .and(AlphabetSamplerSchema)
    .and(PenaltySamplerSchema)
    .and(DrySchema)
    .and(XtcSchema)
    .and(DynatempSchema)
    .and(MirostatSchema);

export type BaseSamplerRequest = z.infer<typeof BaseSamplerRequest>;

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

function getDefaultValue<T>(key: string, fallback: T): T {
    const defaultValue = overridesContainer.overrides[key]?.override ??
        fallback;

    return defaultValue as T;
}
