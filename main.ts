import { createApi } from "@/api/server.ts";
import { loadAuthKeys } from "@/common/auth.ts";
import { parseArgs } from "@/common/args.ts";
import { config, loadConfig } from "@/common/config.ts";
import { logger } from "@/common/logging.ts";
import { applyLoadDefaults, loadModel } from "@/common/modelContainer.ts";
import { elevateProcessPriority, getYalsVersion } from "@/common/utils.ts";
import { overridesFromFile } from "@/common/samplerOverrides.ts";
import { loadYalsBindings } from "@/bindings/lib.ts";
import { ModelConfig } from "./common/configModels.ts";

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

    // Load model if present
    if (config.model.model_name) {
        // Apply load defaults
        // NOTE: inline overrides do not persist across loads
        const initialParams = ModelConfig.parse(
            await applyLoadDefaults(config.model),
        );

        // Load model in bindings
        await loadModel(initialParams);
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
