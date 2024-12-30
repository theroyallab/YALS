import * as z from "@/common/myZod.ts";
import { ModelConfig } from "@/common/configModels.ts";

export const ModelLoadRequest = ModelConfig.extend({
    model_name: z.string(),
}).omit({
    model_dir: true,
});

export const ModelCard = z.object({
    id: z.string().default("test"),
    object: z.string().default("model"),
    created: z.number().default(Date.now()),
    owned_by: z.string().default("YALS"),
});

export type ModelCard = z.infer<typeof ModelCard>;

export const ModelList = z.object({
    object: z.string().default("list"),
    data: z.array(ModelCard).default([]),
});

export type ModelList = z.infer<typeof ModelList>;
