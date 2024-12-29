import * as YAML from "@std/yaml";
import * as z from "@/common/myZod.ts";
import { logger } from "@/common/logging.ts";

const AuthFileSchema = z.object({
    api_key: z.string(),
    admin_key: z.string(),
});

type AuthFile = z.infer<typeof AuthFileSchema>;

export enum AuthKeyPermission {
    API = "API",
    Admin = "Admin",
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

function generateApiToken() {
    const buffer = new Uint8Array(16);
    crypto.getRandomValues(buffer);

    // To hex string
    const token = Array.from(buffer)
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");

    return token;
}

export let authKeys: AuthKeys | undefined = undefined;

export async function loadAuthKeys() {
    const authFilePath = "api_tokens.yml";

    const fileInfo = await Deno.stat(authFilePath).catch(() => null);
    if (fileInfo?.isFile) {
        const rawKeys = await Deno.readTextFile(authFilePath);
        const parsedKeys = await AuthFileSchema.parseAsync(YAML.parse(rawKeys));
        authKeys = new AuthKeys(
            parsedKeys.api_key,
            parsedKeys.admin_key,
        );
    } else {
        const newAuthFile = await AuthFileSchema.parseAsync({
            api_key: generateApiToken(),
            admin_key: generateApiToken(),
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
