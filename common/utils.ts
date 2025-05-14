import os from "node:os";
import { logger } from "./logging.ts";

export function defer(callback: () => void): Disposable {
    return {
        [Symbol.dispose]: () => callback(),
    };
}

export function asyncDefer(callback: () => Promise<void>): AsyncDisposable {
    return {
        [Symbol.asyncDispose]: async () => await callback(),
    };
}

export async function getCommitSha() {
    const cmd = new Deno.Command("git", {
        args: ["rev-parse", "--short", "HEAD"],
    });
    try {
        const { stdout } = await cmd.output();
        const sha = new TextDecoder().decode(stdout).trim();

        return sha;
    } catch (error) {
        console.error(`Failed to get commit SHA: ${error}`);
        return undefined;
    }
}

export async function getYalsVersion(root?: string) {
    const shaPath = root ? `${root}/gitSha.txt` : "gitSha.txt";

    try {
        const cachedSha = await Deno.readTextFile(shaPath);
        return cachedSha.trim();
    } catch {
        return await getCommitSha();
    }
}

export function generateUuidHex() {
    const buffer = new Uint8Array(16);
    crypto.getRandomValues(buffer);

    // To hex string
    const token = Array.from(buffer)
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");

    return token;
}

// Sets the process priority to realtime
export function elevateProcessPriority() {
    try {
        os.setPriority(os.constants.priority.PRIORITY_HIGHEST);
        logger.warn("EXPERIMENTAL: Process priority set to Realtime.");

        if (Deno.build.os === "windows") {
            logger.warn(
                "If you're not running YALS as administrator," +
                    "the priority is set to high.",
            );
        }
    } catch {
        logger.warn(
            "Cannot set the process priority to realtime. " +
                "Restart the program with sudo permissions.",
        );
    }
}
