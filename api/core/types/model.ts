import "zod-openapi/extend";
import { ModelConfig } from "@/common/configModels.ts";

export const ModelLoadRequest = ModelConfig.omit({
    model_dir: true,
});
