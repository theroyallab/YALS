import * as YAML from "@std/yaml";
import * as z from "@/common/myZod.ts";
import { config } from "@/common/config.ts";
import { logger } from "@/common/logging.ts";
import { generateUuidHex } from "@/common/utils.ts";

const AuthFileSchema = z.object({
    api_key: z.string(),
    admin_key: z.string(),
});

type AuthFile = z.infer<typeof AuthFileSchema>;

export enum AuthKeyPermission {
    API = "api",
    Admin = "admin",
}

export class AuthKeys {
    public apiKey: string;
    public adminKey: string;

    public constructor(
        apiKey: string,
        adminKey: string,
    ) {
        this.apiKey = apiKey;
        this.adminKey = adminKey;
    }

    public verifyKey(testKey: string, permission: AuthKeyPermission): boolean {
        switch (permission) {
            case AuthKeyPermission.Admin:
                return testKey === this.adminKey;
            case AuthKeyPermission.API:
                return testKey === this.apiKey || testKey === this.adminKey;
            default:
                return false;
        }
    }
}

export let authKeys: AuthKeys | undefined = undefined;

export async function loadAuthKeys() {
    const authFilePath = "api_tokens.yml";

    if (config.network.disable_auth) {
        logger.warn(
            "Disabling authentication makes your instance vulnerable. \n" +
                "Set the `disable_auth` flag to false in config.yml " +
                "to share this instance with others.",
        );
    }

    const fileInfo = await Deno.stat(authFilePath).catch(() => null);
    if (fileInfo?.isFile) {
        const rawKeys = await Deno.readTextFile(authFilePath);
        const parsedKeys = AuthFileSchema.parse(YAML.parse(rawKeys));
        authKeys = new AuthKeys(
            parsedKeys.api_key,
            parsedKeys.admin_key,
        );
    } else {
        const newAuthFile = AuthFileSchema.parse({
            api_key: generateUuidHex(),
            admin_key: generateUuidHex(),
        });

        authKeys = new AuthKeys(
            newAuthFile.api_key,
            newAuthFile.admin_key,
        );

        await Deno.writeFile(
            authFilePath,
            new TextEncoder().encode(YAML.stringify(newAuthFile)),
        );
    }

    logger.info(
        "\n" +
            `Your API key is: ${authKeys.apiKey}\n` +
            `Your Admin key is: ${authKeys.adminKey}\n\n` +
            "If these keys get compromised, make sure to delete api_tokens.yml " +
            "and restart the server. Have fun!",
    );
}

export function getAuthPermission(
    headers: Record<string, string>,
): AuthKeyPermission {
    if (config.network.disable_auth) {
        return AuthKeyPermission.Admin;
    }

    let testKey = headers["x-admin-key"] ?? headers["x-api-key"] ??
        headers["authorization"];

    if (!testKey) {
        throw new Error("The provided authentication key is missing.");
    }

    if (testKey.toLowerCase().startsWith("bearer")) {
        testKey = testKey.split(" ")[1];
    }

    if (authKeys?.verifyKey(testKey, AuthKeyPermission.Admin)) {
        return AuthKeyPermission.Admin;
    } else if (authKeys?.verifyKey(testKey, AuthKeyPermission.API)) {
        return AuthKeyPermission.API;
    } else {
        throw new Error("The provided authentication key is invalid.");
    }
}
