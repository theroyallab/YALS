import {
    configure,
    getAnsiColorFormatter,
    getConsoleSink,
    getLogger,
} from "logtape";
import { config } from "@/common/config.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";

export async function setupLogger() {
    const formatter = getAnsiColorFormatter({
        level: "FULL",
        timestamp: "date-time",
        format: (values) =>
            `${values.timestamp} ${values.level} ${values.category}: ${values.message}`,
    });

    await configure({
        sinks: { console: getConsoleSink({ formatter: formatter }) },
        loggers: [
            { category: "YALS", level: "debug", sinks: ["console"] },
            { category: ["logtape", "meta"], level: "error" },
        ],
    });
}

export const logger = getLogger("YALS");

export function logPrompt(prompt: string) {
    // Log prompt to console
    if (config.logging.log_prompt) {
        logger.info(
            "Prompt: \n" + prompt,
        );
    }
}

export function logGenParams(params: BaseSamplerRequest) {
    if (config.logging.log_generation_params) {
        const samplerParams = BaseSamplerRequest.parse(params);
        const formattedParams = Deno.inspect(samplerParams, {
            depth: 2,
            compact: true,
            breakLength: Infinity,
        });

        logger.info("Generation Parameters: {formattedParams}", {
            formattedParams,
        });
    }
}
