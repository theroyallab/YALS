import { createApi } from "@/api/server.ts";
import { setupLogger } from "@/common/logging.ts";
import { loadModel } from "@/common/modelContainer.ts";
import { basename, parse } from "@std/path";

if (import.meta.main) {
    const parsedPath = parse(
        "D:/koboldcpp/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
    );
    await setupLogger();
    await loadModel("D:/koboldcpp/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf", 999);
    createApi();
}
