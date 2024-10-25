import { setupBindings } from "./bindings/bindings.ts";

// Learn more at https://docs.deno.com/runtime/manual/examples/module_metadata#concepts
if (import.meta.main) {
    await setupBindings();
}
