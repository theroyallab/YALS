import * as z from "@/common/myZod.ts";

export const TemplateList = z.object({
    object: z.string().default("list"),
    data: z.array(z.string()).default([]),
});

export const TemplateSwitchRequest = z.aliasedObject(
    z.object({
        prompt_template_name: z.string(),
    }),
    [{ field: "prompt_template_name", aliases: ["name"] }],
);
