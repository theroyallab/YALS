import { createMiddleware } from "hono/factory";
import { logger } from "../../common/logging.ts";

// Middleware for logging parts of a request
const requestLogMiddleware = createMiddleware(
    async (c, next) => {
        const logMessage = [
            `Information for ${c.req.method} request ${c.var.requestId}`,
        ];

        logMessage.push(`URL: ${c.req.url}`);

        const headers = Object.fromEntries(c.req.raw.headers);
        logMessage.push(`Headers: ${JSON.stringify(headers, null, 2)}`);

        if (c.req.method !== "GET") {
            const clonedReq = c.req.raw.clone();
            const textBody = await clonedReq.text();

            if (textBody) {
                logMessage.push(`Body: ${textBody}`);
            }
        }

        logger.info(logMessage.join("\n"));

        await next();
    },
);

export default requestLogMiddleware;
