import * as z from "zod";

// Coalesce function

function coalesce<T extends z.ZodType, D extends NonNullable<z.output<T>>>(
    this: T,
    defaultValue: D,
) {
    // Needs a second assertion to satisfy arrays
    return this.transform((val) =>
        val ?? (defaultValue as NonNullable<z.output<T>>)
    );
}

z.ZodType.prototype.coalesce = coalesce;

// cleanOptional

// Coerces a nullish value to undefined
// This allows null + undefined to be accepted
// but follows TS's rule of undefined being an absence of a value
function cleanOptional<T extends z.ZodType>(
    this: T,
) {
    return this.nullish().transform((val) => val ?? undefined);
}

z.ZodType.prototype.cleanOptional = cleanOptional;

// Sampler overrides

// Store the sampler override default function to prevent circular import
let samplerOverrideResolver = (_key: string): unknown | null | undefined =>
    undefined;

export function registerSamplerOverrideResolver(
    resolver: (key: string) => unknown | null | undefined,
) {
    samplerOverrideResolver = resolver;
}

const samplerOverride = function <T extends z.ZodType>(this: T, key: string) {
    return this.transform((value, ctx) => {
        if (value !== undefined && value !== null) {
            return value;
        }

        const defaultValue = samplerOverrideResolver(key);
        const result = this.safeParse(defaultValue);
        if (result.success) {
            return defaultValue as z.output<T>;
        } else {
            let expectedType = "";

            const issues = result.error.issues;
            if (issues.length > 0 && issues[0].code === "invalid_type") {
                const issue = issues[0] as z.core.$ZodIssueInvalidType;
                expectedType = issue.expected;
            }

            ctx.addIssue({
                code: "custom",
                message: `Sampler override for ${key} must match ` +
                    `the input type ${expectedType}`,
                input: defaultValue,
                path: ["samplerOverride"],
            });

            return z.NEVER;
        }
    });
};

z.ZodType.prototype.samplerOverride = samplerOverride;

// Alias support
interface AliasChoice {
    field: string;
    aliases: string[];
}

export function aliasedObject<
    O extends z.ZodTypeAny,
>(
    schema: O,
    aliasChoices: AliasChoice[],
) {
    return z.preprocess((item: unknown) => {
        const obj = z.record(z.string(), z.unknown()).safeParse(item);
        if (obj.success) {
            for (const choice of aliasChoices) {
                // If the field contains a value, skip
                if (obj.data[choice.field]) {
                    continue;
                }

                // Replace with the first found alias value
                const foundAlias = choice.aliases.find((alias) =>
                    alias in obj.data
                );
                if (foundAlias) {
                    obj.data[choice.field] = obj.data[foundAlias];
                }
            }

            // Reassign the object
            item = obj.data;
        }

        return obj.data;
    }, schema);
}

// Export all types
export * from "zod";

declare module "zod" {
    interface ZodType {
        coalesce<D extends NonNullable<z.output<this>>>(
            defaultValue: D,
        ): ReturnType<typeof coalesce<this, D>>;

        cleanOptional(): ReturnType<typeof cleanOptional<this>>;

        samplerOverride(
            key: string,
        ): ReturnType<typeof samplerOverride<this>>;
    }
}
