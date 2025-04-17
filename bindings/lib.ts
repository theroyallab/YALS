import libraryInterface from "./symbols.ts";

export let lib: Deno.DynamicLibrary<typeof libraryInterface>;
export let hasLlguidance: boolean = false;

export function loadYalsBindings() {
    const libName = "c_library";
    const libDir = `${Deno.cwd()}/lib/`;
    let libPath = libDir;

    switch (Deno.build.os) {
        case "windows":
            Deno.env.set("PATH", `${Deno.env.get("PATH")};${libDir}`);
            libPath += `${libName}.dll`;
            break;
        case "linux":
            libPath += `lib${libName}.so`;
            break;
        case "darwin":
            libPath += `lib${libName}.dylib`;
            break;
        default:
            throw new Error(`Unsupported operating system: ${Deno.build.os}`);
    }

    try {
        lib = Deno.dlopen(libPath, libraryInterface);
        hasLlguidance = lib.symbols.has_llguidance();
    } catch (error: unknown) {
        console.error(
            `Failed to load YALS library: ${
                error instanceof Error ? error.message : String(error)
            }`,
        );
        throw error;
    }
}
