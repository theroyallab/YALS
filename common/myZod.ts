import * as z from "zod";
import "zod-openapi/extend";
export * from "zod";

// Extend ZodType
z.ZodType.prototype.coalesce = function (defaultValue) {
    return this.transform((value) => value ?? defaultValue);
};

// Export type as part of the package
declare module "zod" {
    interface ZodType<
        Output,
        Def extends z.ZodTypeDef = z.ZodTypeDef,
        Input = Output,
    > {
        coalesce(
            defaultValue: NonNullable<Output>,
        ): z.ZodEffects<this, NonNullable<Output>>;
    }
}

// Alias support
interface AliasChoice {
    field: string;
    aliases: string[];
}

export function aliasedObject<
    O extends z.ZodObject<S>,
    S extends z.ZodRawShape,
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
