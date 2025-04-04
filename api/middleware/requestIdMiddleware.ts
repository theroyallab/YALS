import { createMiddleware } from "hono/factory";
import { generateUuidHex } from "@/common/utils.ts";

// Extra global vars for context
declare module "hono" {
    export interface HonoRequest {
        id: string;
    }
}

// Assigns an ID to a request
const requestIdMiddleware = createMiddleware(
    async (c, next) => {
        c.req.id = generateUuidHex();

        await next();
    },
);

export default requestIdMiddleware;
