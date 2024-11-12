import { createApi } from "@/api/server.ts";
import { setupLogger } from "@/common/logging.ts";
import { loadModel } from "@/common/modelContainer.ts";

if (import.meta.main) {
<<<<<<< Updated upstream
=======
    const parsedPath = parse(
        "/home/blackroot/Desktop/YALS/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    );
>>>>>>> Stashed changes
    await setupLogger();
    await loadModel("/home/blackroot/Desktop/YALS/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", 999);
    createApi();
}
