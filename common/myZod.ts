import * as z from "zod";
import "zod-openapi/extend";

// Extend ZodType

// Coalesce to handle nullish values
// If the default is a function, call it and return
z.ZodType.prototype.coalesce = function (defaultValue) {
    return this.transform((value) => {
        if (value != null) {
            return value;
        } else if (typeof defaultValue === "function") {
            return defaultValue();
        } else {
            return defaultValue;
        }
    });
};

// Sampler overrides

// Store the sampler override default function to prevent circular import
let samplerOverrideResolver = <T>(_key: string): T | null | undefined =>
    undefined;

export function registerSamplerOverrideResolver(
    resolver: <T>(key: string) => T | null | undefined,
) {
    samplerOverrideResolver = resolver;
}

z.ZodType.prototype.samplerOverride = function <T>(key: string) {
    return this.transform((_value) => samplerOverrideResolver<T>(key));
};

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
        const obj = z.record(z.unknown()).safeParse(item);
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

export * from "zod";

// Export type as part of the package
declare module "zod" {
    interface ZodType<
        Output,
        Def extends z.ZodTypeDef = z.ZodTypeDef,
        Input = Output,
    > {
        coalesce(
            defaultValue: NonNullable<Output> | (() => NonNullable<Output>),
        ): z.ZodEffects<this, NonNullable<Output>>;

        samplerOverride<_T>(key: string): z.ZodEffects<this, Output>;
    }
}
