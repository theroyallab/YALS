import "zod-openapi/extend";
import { z } from "zod";

const maxTokensType = z.number().gte(0).nullish();
const banEosTokenType = z.boolean().nullish();
const stopStringsType = z.union([
    z.string(),
    z.array(z.string()),
]).nullish();
const bannedTokensType = z.union([
    z.array(z.number()),
    z.string(),
]).nullish();
const GenerationOptionsSchema = z.object({
    grammar_string: z.string().nullish(),
    add_bos_token: z.boolean().nullish().coalesce(true),
    skip_special_tokens: z.boolean().nullish().coalesce(false),
    seed: z.number().nullish(),
    logit_bias: z.record(z.string(), z.number()).nullish().coalesce({}),
    banned_strings: z.union([z.string(), z.array(z.string())]).nullish()
        .coalesce([]),

    // max token aliases
    max_tokens: maxTokensType
        .openapi({
            description: "Aliases: max_length",
        }),
    max_length: maxTokensType,

    // ban_eos_token aliases
    ban_eos_token: banEosTokenType
        .openapi({
            description: "Aliases: ignore_eos",
        }),
    ignore_eos: banEosTokenType,

    // stop aliases
    stop: stopStringsType
        .openapi({
            description: "Aliases: stop_sequence",
        }),
    stop_sequence: stopStringsType,

    // banned_tokens aliases
    banned_tokens: bannedTokensType
        .openapi({
            description: "Aliases: custom_token_bans",
        }),
    custom_token_bans: bannedTokensType,
})
    .openapi({
        description: "Generation options",
    })
    .transform((obj) => {
        // Aliases
        const aliasObj = {
            ...obj,
            max_tokens: obj.max_tokens ?? obj.max_length,
            ban_eos_token: obj.ban_eos_token ?? obj.ignore_eos ?? false,
            stop: obj.stop ?? obj.stop_sequence ?? [],
            banned_tokens: obj.banned_tokens ?? obj.custom_token_bans ?? [],
        };

        // TODO: Possibly fix redundancy?

        // Transform associated values
        if (typeof aliasObj.stop == "string") {
            aliasObj.stop = [aliasObj.stop];
        }

        if (typeof aliasObj.banned_tokens == "string") {
            aliasObj.banned_tokens = aliasObj.banned_tokens.replaceAll(" ", "")
                .split(",")
                .filter((x) => /^\d+$/.test(x))
                .map((x) => parseInt(x));
        }

        if (typeof aliasObj.banned_strings == "string") {
            aliasObj.banned_strings = [aliasObj.banned_strings];
        }

        // Cast the transformed types
        return {
            ...aliasObj,
            stop: aliasObj.stop as string[],
            banned_tokens: aliasObj.banned_tokens as number[],
            banned_strings: aliasObj.banned_strings as string[],
        };
    });

const TemperatureSamplerSchema = z.object({
    temperature: z.number().gte(0).nullish().coalesce(1),
    temperature_last: z.boolean().nullish().coalesce(false),
})
    .openapi({
        description: "Temperature options",
    });

const typicalType = z.number().gt(0).lte(1).nullish();
const AlphabetSamplerSchema = z.object({
    top_k: z.number().gte(-1).default(0),
    top_p: z.number().gte(0).lte(1).nullish().coalesce(1),
    min_p: z.number().gte(0).lte(1).nullish().coalesce(0),

    // typical aliases
    typical: typicalType,
    typical_p: typicalType,
})
    .openapi({
        description: "Alphabet samplers",
    })
    .transform((obj) => {
        // Aliases
        const aliasObj = {
            ...obj,
            typical: obj.typical ?? obj.typical_p ?? 1,
        };

        // Transform associated values
        if (aliasObj.top_k == -1) {
            aliasObj.top_k = 0;
        }

        return aliasObj;
    });

const repetitionPenaltyType = z.number().gt(0).nullish();
const penaltyRangeType = z.number().nullish();
const PenaltySamplerSchema = z.object({
    frequency_penalty: z.number().gte(0).nullish().coalesce(0),
    presence_penalty: z.number().gte(0).nullish().coalesce(0),

    // repetition_penalty aliases
    repetition_penalty: repetitionPenaltyType
        .openapi({
            description: "Aliases: rep_pen",
        }),
    rep_pen: repetitionPenaltyType,

    // penalty_range aliases
    penalty_range: penaltyRangeType
        .openapi({
            description:
                "Aliases: repetition_range, repetition_penalty_range, rep_pen_range",
        }),
    repetition_range: penaltyRangeType,
    repetition_penalty_range: penaltyRangeType,
    rep_pen_range: penaltyRangeType,
})
    .openapi({
        description: "Penalty samplers",
    })
    .transform((obj) => {
        return {
            ...obj,
            repetition_penalty: obj.repetition_penalty ?? obj.rep_pen ?? 1,
            penalty_range: obj.penalty_range ??
                obj.repetition_range ??
                obj.repetition_penalty_range ??
                obj.rep_pen_range ??
                -1,
        };
    });

const dryRangeType = z.number().nullish();
const DrySchema = z.object({
    dry_multiplier: z.number().nullish().coalesce(0),
    dry_base: z.number().nullish().coalesce(0),
    dry_allowed_length: z.number().nullish().coalesce(0),
    dry_sequence_breakers: z.union([z.string(), z.array(z.string())]).nullish()
        .coalesce(
            [],
        ),

    // dry_range aliases
    dry_range: dryRangeType
        .openapi({
            description: "Aliases: dry_penalty_last_n",
        }),
    dry_penalty_last_n: dryRangeType,
})
    .openapi({
        description: "DRY options",
    })
    .transform((obj) => {
        // Aliases
        const aliasObj = {
            ...obj,
            dry_range: obj.dry_range ?? obj.dry_penalty_last_n ?? 0,
        };

        // Transform associated values
        if (typeof aliasObj.dry_sequence_breakers == "string") {
            if (!aliasObj.dry_sequence_breakers.startsWith("[")) {
                aliasObj.dry_sequence_breakers =
                    `[${aliasObj.dry_sequence_breakers}]`;
            }

            // Parse can fail, so return a default value if it does
            try {
                aliasObj.dry_sequence_breakers = JSON.parse(
                    aliasObj.dry_sequence_breakers,
                );
            } catch {
                aliasObj.dry_sequence_breakers = [];
            }
        }

        // Cast the transformed types
        return {
            ...aliasObj,
            dry_sequence_breakers: aliasObj.dry_sequence_breakers as string[],
        };
    });

const XtcSchema = z.object({
    xtc_probability: z.number().nullish().coalesce(0),
    xtc_threshold: z.number().nullish().coalesce(0.1),
})
    .openapi({
        description: "XTC options",
    });

const maxTempType = z.number().gte(0).nullish();
const minTempType = z.number().gte(0).nullish();
const tempExponentType = z.number().gte(0).nullish();
const DynatempSchema = z.object({
    // Aliases for max_temp
    max_temp: maxTempType
        .openapi({
            description: "Aliases: dynatemp_high",
        }),
    dynatemp_high: maxTempType,

    // Aliases for min_temp
    min_temp: minTempType
        .openapi({
            description: "Aliases: dynatemp_low",
        }),
    dynatemp_low: minTempType,

    // Aliases for temp_exponent
    temp_exponent: tempExponentType
        .openapi({
            description: "Aliases: dynatemp_exponent",
        }),
    dynatemp_exponent: tempExponentType,
})
    .openapi({
        description: "DynaTemp options",
    })
    .transform((obj) => {
        return {
            min_temp: obj.min_temp ?? obj.dynatemp_low ?? 1,
            max_temp: obj.max_temp ?? obj.dynatemp_high ?? 1,
            temp_exponent: obj.temp_exponent ?? obj.dynatemp_exponent ?? 1,
        };
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
