import "zod-openapi/extend";
import { z } from "zod";

const maxTokensType = z.number().gte(0).optional();
const banEosTokenType = z.boolean().optional();
const stopStringsType = z.union([
    z.string(),
    z.array(z.union([z.number(), z.string()])),
]).optional();
const bannedTokensType = z.union([
    z.array(z.number()),
    z.string(),
]).optional();
const GenerationOptionsSchema = z.object({
    grammar_string: z.string().optional(),
    add_bos_token: z.boolean().default(true),
    skip_special_tokens: z.boolean().default(true),
    seed: z.number().optional(),
    logit_bias: z.record(z.string(), z.number()).default({}),

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
        "description": "Generation options",
    })
    .transform((obj) => {
        return {
            ...obj,
            max_tokens: obj.max_tokens ?? obj.max_length,
            ban_eos_token: obj.ban_eos_token ?? obj.ignore_eos ?? false,
            stop: obj.stop ?? obj.stop_sequence ?? [],
            banned_tokens: obj.banned_tokens ?? obj.custom_token_bans ?? [],
        };
    });

const TemperatureSamplerSchema = z.object({
    temperature: z.number().gte(0).default(1),
    temperature_last: z.boolean().default(false),
})
    .openapi({
        description: "Temperature options",
    });

const typicalType = z.number().gt(0).lte(1).optional();
const AlphabetSamplerSchema = z.object({
    top_k: z.number().gte(-1).default(0),
    top_p: z.number().gte(0).lte(1).default(1),
    min_p: z.number().gte(0).lte(1).default(0),

    // typical aliases
    typical: typicalType,
    typical_p: typicalType,
})
    .openapi({
        description: "Alphabet samplers",
    })
    .transform((obj) => {
        return {
            ...obj,
            typical: obj.typical ?? obj.typical_p ?? 1,
        };
    });

const repetitionPenaltyType = z.number().gt(0).optional();
const PenaltySamplerSchema = z.object({
    frequency_penalty: z.number().gte(0).default(0),
    presence_penalty: z.number().gte(0).default(0),

    // repetition_penalty aliases
    repetition_penalty: repetitionPenaltyType
        .openapi({
            description: "Aliases: rep_pen",
        }),
    rep_pen: repetitionPenaltyType,
})
    .openapi({
        description: "Penalty samplers",
    })
    .transform((obj) => {
        return {
            ...obj,
            repetition_penalty: obj.repetition_penalty ?? obj.rep_pen ?? 1,
        };
    });

const dryRangeType = z.number().optional();
const DrySchema = z.object({
    dry_multiplier: z.number().default(0),
    dry_base: z.number().default(0),
    dry_allowed_length: z.number().default(0),
    dry_sequence_breakers: z.union([z.string(), z.array(z.string())]).default(
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
        return {
            ...obj,
            dry_range: obj.dry_range ?? obj.dry_penalty_last_n ?? 0,
        };
    });

const XtcSchema = z.object({
    xtc_probability: z.number().default(0),
    xtc_threshold: z.number().default(0.1),
})
    .openapi({
        description: "XTC options",
    });

const maxTempType = z.number().gte(0).optional();
const minTempType = z.number().gte(0).optional();
const tempExponentType = z.number().gte(0).optional();
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
    mirostat: z.boolean().default(false),
    mirostat_mode: z.number().default(0),
    mirostat_tau: z.number().default(1),
    mirostat_eta: z.number().default(0),
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
