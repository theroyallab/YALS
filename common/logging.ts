import {
    configure,
    getAnsiColorFormatter,
    getConsoleSink,
    getLogger,
} from "logtape";

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
