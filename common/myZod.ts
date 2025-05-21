import * as z from "zod/v4";

// Coalesce function

function coalesce<T extends z.ZodType, D>(this: T, defaultValue: D) {
    return this
        .transform((val) => val ?? defaultValue)
        .refine((val) => val !== undefined && val !== null, {
            error: "Coalesced value cannot be undefined or null",
        });
}

z.ZodType.prototype.coalesce = coalesce;

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
    return this.transform((value) => {
        if (value !== undefined && value !== null) {
            return value;
        }

        const defaultValue = samplerOverrideResolver(key);

        // Make sure the default value adheres to the type
        return this.parse(defaultValue);
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
export * from "zod/v4";

declare module "zod/v4" {
    interface ZodType {
        coalesce<T extends ZodType, D>(
            this: T,
            defaultValue: D,
        ): ReturnType<typeof coalesce<T, D>>;

        samplerOverride<T extends ZodType>(
            this: T,
            key: string,
        ): ReturnType<typeof samplerOverride<T>>;
    }
}
