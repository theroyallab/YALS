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

export async function getYalsVersion() {
    try {
        return Deno.readTextFileSync(`${import.meta.dirname}/gitSha.txt`)
            .trim();
    } catch {
        return await getCommitSha();
    }
}
