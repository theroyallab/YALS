import * as z from "@/common/myZod.ts";

export const HealthSchema = z.object({
    health: z.enum(["ok", "unhealthy"]),
});
