import * as v from "valibot";

const maxTokensType = v.nullish(v.pipe(v.number(), v.minValue(0)));
const banEosTokenType = v.nullish(v.boolean());
const stopStringsType = v.nullish(v.union([
    v.string(),
    v.array(v.string()),
]));
const bannedTokensType = v.nullish(v.union([
    v.array(v.number()),
    v.string(),
]));
const GenerationOptionsSchema = v.pipe(
    v.object({
        grammar_string: v.nullish(v.string()),
        add_bos_token: v.nullish(v.boolean(), true),
        skip_special_tokens: v.nullish(v.boolean(), false),
        seed: v.nullish(v.number()),
        logit_bias: v.nullish(v.record(v.string(), v.number()), {}),
        banned_strings: v.nullish(
            v.union([v.string(), v.array(v.string())]),
            [],
        ),

        // max token aliases
        max_tokens: v.pipe(maxTokensType, v.description("Aliases: max_length")),
        max_length: maxTokensType,

        // ban_eos_token aliases
        ban_eos_token: v.pipe(
            banEosTokenType,
            v.description("Aliases: ignore_eos"),
        ),
        ignore_eos: banEosTokenType,

        // stop aliases
        stop: v.pipe(stopStringsType, v.description("Aliases: stop_sequence")),
        stop_sequence: stopStringsType,

        // banned_tokens aliases
        banned_tokens: v.pipe(
            bannedTokensType,
            v.description("Aliases: custom_token_bans"),
        ),
        custom_token_bans: bannedTokensType,
    }),
    v.description("Generation options"),
    v.transform((obj) => {
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
    }),
);

const TemperatureSamplerSchema = v.pipe(
    v.object({
        temperature: v.nullish(v.pipe(v.number(), v.minValue(0)), 1),
        temperature_last: v.nullish(v.boolean(), false),
    }),
    v.description("Temperature options"),
);

const typicalType = v.nullish(
    v.pipe(v.number(), v.minValue(0), v.notValue(0), v.maxValue(1)),
);
const AlphabetSamplerSchema = v.pipe(
    v.object({
        top_k: v.nullish(v.pipe(v.number(), v.minValue(-1)), 0),
        top_p: v.nullish(v.pipe(v.number(), v.minValue(0), v.maxValue(1)), 1),
        min_p: v.nullish(v.pipe(v.number(), v.minValue(0), v.maxValue(1)), 0),

        // typical aliases
        typical: typicalType,
        typical_p: typicalType,
    }),
    v.description("Alphabet samplers"),
    v.transform((obj) => {
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
    }),
);

const repetitionPenaltyType = v.nullish(
    v.pipe(v.number(), v.minValue(0), v.notValue(0)),
);
const penaltyRangeType = v.nullish(v.number());
const PenaltySamplerSchema = v.pipe(
    v.object({
        frequency_penalty: v.nullish(v.pipe(v.number(), v.minValue(0)), 0),
        presence_penalty: v.nullish(v.pipe(v.number(), v.minValue(0)), 0),

        // repetition_penalty aliases
        repetition_penalty: v.pipe(
            repetitionPenaltyType,
            v.description("Aliases: rep_pen"),
        ),
        rep_pen: repetitionPenaltyType,

        // penalty_range aliases
        penalty_range: v.pipe(
            penaltyRangeType,
            v.description(
                "Aliases: repetition_range, repetition_penalty_range, rep_pen_range",
            ),
        ),
        repetition_range: penaltyRangeType,
        repetition_penalty_range: penaltyRangeType,
        rep_pen_range: penaltyRangeType,
    }),
    v.description("Penalty samplers"),
    v.transform((obj) => {
        return {
            ...obj,
            repetition_penalty: obj.repetition_penalty ?? obj.rep_pen ?? 1,
            penalty_range: obj.penalty_range ??
                obj.repetition_range ??
                obj.repetition_penalty_range ??
                obj.rep_pen_range ??
                -1,
        };
    }),
);

const dryRangeType = v.nullish(v.number());
const DrySchema = v.pipe(
    v.object({
        dry_multiplier: v.nullish(v.number(), 0),
        dry_base: v.nullish(v.number(), 0),
        dry_allowed_length: v.nullish(v.number(), 0),
        dry_sequence_breakers: v.nullish(
            v.union([v.string(), v.array(v.string())]),
            [],
        ),

        // dry_range aliases
        dry_range: v.pipe(
            dryRangeType,
            v.description("Aliases: dry_penalty_last_n"),
        ),
        dry_penalty_last_n: dryRangeType,
    }),
    v.description("DRY options"),
    v.transform((obj) => {
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
    }),
);

const XtcSchema = v.pipe(
    v.object({
        xtc_probability: v.nullish(v.number(), 0),
        xtc_threshold: v.nullish(v.number(), 0.1),
    }),
    v.description("XTC options"),
);

const maxTempType = v.nullish(v.pipe(v.number(), v.minValue(0)));
const minTempType = v.nullish(v.pipe(v.number(), v.minValue(0)));
const tempExponentType = v.nullish(v.pipe(v.number(), v.minValue(0)));
const DynatempSchema = v.pipe(
    v.object({
        // Aliases for max_temp
        max_temp: v.pipe(maxTempType, v.description("Aliases: dynatemp_high")),
        dynatemp_high: maxTempType,

        // Aliases for min_temp
        min_temp: v.pipe(minTempType, v.description("Aliases: dynatemp_low")),
        dynatemp_low: minTempType,

        // Aliases for temp_exponent
        temp_exponent: v.pipe(
            tempExponentType,
            v.description("Aliases: dynatemp_exponent"),
        ),
        dynatemp_exponent: tempExponentType,
    }),
    v.description("DynaTemp options"),
    v.transform((obj) => {
        return {
            min_temp: obj.min_temp ?? obj.dynatemp_low ?? 1,
            max_temp: obj.max_temp ?? obj.dynatemp_high ?? 1,
            temp_exponent: obj.temp_exponent ?? obj.dynatemp_exponent ?? 1,
        };
    }),
);

const MirostatSchema = v.pipe(
    v.object({
        mirostat_mode: v.nullish(v.number(), 0),
        mirostat_tau: v.nullish(v.number(), 1),
        mirostat_eta: v.nullish(v.number(), 0),
    }),
    v.description("Mirostat options"),
);

export const BaseSamplerRequest = v.intersect([
    GenerationOptionsSchema,
    TemperatureSamplerSchema,
    AlphabetSamplerSchema,
    PenaltySamplerSchema,
    DrySchema,
    XtcSchema,
    DynatempSchema,
    MirostatSchema,
]);

export type BaseSamplerRequest = v.InferOutput<typeof BaseSamplerRequest>;
