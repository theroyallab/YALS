import { HTTPException } from "hono/http-exception";
import { createMiddleware } from "hono/factory";
import { AuthKeyPermission, authKeys } from "@/common/auth.ts";
import { config } from "@/common/config.ts";

// Middleware for key validation
const authMiddleware = (permission: AuthKeyPermission) => {
    return createMiddleware(async (c, next) => {
        if (config.network.disable_auth) {
            await next();
            return;
        }

        const headers = c.req.header();
        const xHeader = `x-${permission.toLowerCase()}-key`;

        // TODO: Possibly refactor error throws
        if (xHeader in headers) {
            const valid = authKeys?.verifyKey(headers[xHeader], permission);
            if (!valid) {
                throw new HTTPException(401, {
                    message: `Invalid ${permission} key`,
                });
            }
        } else if ("authorization" in headers) {
            const splitKey = headers["authorization"].split(" ");
            if (splitKey.length < 2) {
                throw new HTTPException(401, {
                    message: `Invalid ${permission} key`,
                });
            }

            const valid = splitKey[0].toLowerCase() === "bearer" &&
                authKeys?.verifyKey(splitKey[1], permission);

            if (!valid) {
                throw new HTTPException(401, {
                    message: `Invalid ${permission} key`,
                });
            }
        } else {
            throw new HTTPException(401, { message: "Key not provided" });
        }

        await next();
    });
};

export default authMiddleware;
