import "zod-openapi/extend";
import { ModelConfig } from "@/common/configModels.ts";
import { z } from "zod";

export const ModelLoadRequest = ModelConfig.omit({
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
