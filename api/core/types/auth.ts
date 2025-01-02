import * as z from "@/common/myZod.ts";

export const AuthPermissionResponse = z.object({
    permission: z.string(),
});
