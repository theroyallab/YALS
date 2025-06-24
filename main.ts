import { createApi } from "@/api/server.ts";
import { loadYalsBindings } from "@/bindings/lib.ts";
import { runAction } from "@/common/actions.ts";
import { loadAuthKeys } from "@/common/auth.ts";
import { parseArgs } from "@/common/args.ts";
import { config, loadConfig } from "@/common/config.ts";
import { logger } from "@/common/logging.ts";
import { loadModel } from "@/common/modelContainer.ts";
import { elevateProcessPriority, getYalsVersion } from "@/common/utils.ts";
import { overridesFromFile } from "@/common/samplerOverrides.ts";

if (import.meta.main) {
    // Use Promise resolution to avoid nested try/catch
    const version = await getYalsVersion(import.meta.dirname);

    if (version) {
        logger.info(`Using YALS commit ${version}`);
    } else {
        logger.info("Could not find YALS commit version. Launching anyway.");
    }

    // Load bindings
    loadYalsBindings();

    // Parse CLI args
    const { args, usage } = parseArgs();

    // Display help message if needed
    if (args.support.help) {
        console.log(usage);
        Deno.exit();
    }

    await loadConfig(args);

    // Defer to an action if specified in invocation
    const ranAction = await runAction(args);
    if (ranAction) {
        Deno.exit();
    }

    // Load model if present
    if (config.model.model_name) {
        await loadModel(config.model);
    }

    // Attempt to set RT process priority
    if (config.developer.realtime_process_priority) {
        elevateProcessPriority();
    }

    // Set sampler overrides
    if (config.sampling.override_preset) {
        await overridesFromFile(config.sampling.override_preset);
    }

    await loadAuthKeys();
    createApi();
}
