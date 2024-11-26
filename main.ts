import { createApi } from "@/api/server.ts";
import { setupLogger } from "@/common/logging.ts";
import { loadModel } from "@/common/modelContainer.ts";
import { config, loadConfig } from "@/common/config.ts";

if (import.meta.main) {
    await setupLogger();
    await loadConfig();

    if (config.model.model_name) {
        await loadModel(config.model);
    }

    createApi();
}
