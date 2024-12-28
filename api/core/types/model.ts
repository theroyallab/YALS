import * as v from "valibot";
import { ModelConfig } from "@/common/configModels.ts";

export const ModelLoadRequest = v.omit(ModelConfig, ["model_dir"]);

export const ModelCard = v.object({
    id: v.nullish(v.string(), "test"),
    object: v.nullish(v.string(), "model"),
    created: v.nullish(v.number(), Date.now()),
    owned_by: v.nullish(v.string(), "YALS"),
});

export type ModelCard = v.InferOutput<typeof ModelCard>;

export const ModelList = v.object({
    object: v.fallback(v.string(), "list"),
    data: v.array(ModelCard),
});

export type ModelList = v.InferOutput<typeof ModelList>;
