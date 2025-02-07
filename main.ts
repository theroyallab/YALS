import { createApi } from "@/api/server.ts";
import { loadAuthKeys } from "@/common/auth.ts";
import { parseArgs } from "@/common/args.ts";
import { config, loadConfig } from "@/common/config.ts";
import { setupLogger } from "@/common/logging.ts";
import { loadModel } from "@/common/modelContainer.ts";

if (import.meta.main) {
    await setupLogger();

    // Parse CLI args
    const { args, usage } = parseArgs();

    // Display help message if needed
    if (args.support.help) {
        console.log(usage);
        Deno.exit();
    }

    await loadConfig(args);

    if (config.model.model_name) {
        await loadModel(config.model);
    }

    await loadAuthKeys();
    createApi();
}
