import { ZodEffects, ZodType } from "zod";

// Extend ZodType
ZodType.prototype.coalesce = function (defaultValue) {
    return this.transform((value) => value ?? defaultValue);
};

// Export type as part of the package
declare module "zod" {
    interface ZodType<
        Output,
        Def extends ZodTypeDef = ZodTypeDef,
        Input = Output,
    > {
        coalesce(
            defaultValue: NonNullable<Output>,
        ): ZodEffects<this, NonNullable<Output>>;
    }
}
