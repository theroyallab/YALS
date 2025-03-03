import { createApi } from "@/api/server.ts";
import { loadAuthKeys } from "@/common/auth.ts";
import { parseArgs } from "@/common/args.ts";
import { config, loadConfig } from "@/common/config.ts";
import { logger, setupLogger } from "@/common/logging.ts";
import { loadModel } from "@/common/modelContainer.ts";
import { getYalsVersion } from "@/common/utils.ts";

if (import.meta.main) {
    await setupLogger();

    // Use Promise resolution to avoid nested try/catch
    const version = await getYalsVersion();

    if (version) {
        logger.info(`Using YALS commit ${version}`);
    } else {
        logger.info("Could not find YALS commit version. Launching anyway.");
    }

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
