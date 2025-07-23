import { lib } from "@/bindings/lib.ts";
import { ReadbackBuffer } from "./readbackBuffer.ts";
import { ReadbackGenerationChunk } from "./types.ts";

export class Job {
    // Private references
    private readbackBuffer: ReadbackBuffer;
    private processor: Deno.PointerValue;

    isComplete = false;
    id: number;

    constructor(
        id: number,
        readbackBuffer: ReadbackBuffer,
        processor: Deno.PointerValue,
    ) {
        this.id = id;
        this.readbackBuffer = readbackBuffer;
        this.processor = processor;
    }

    async *stream(): AsyncGenerator<ReadbackGenerationChunk> {
        for await (const readbackChunk of this.readbackBuffer.read()) {
            yield readbackChunk;
        }

        const status = await this.readbackBuffer.readStatus();
        if (status) {
            yield status;
        }
    }

    cancel() {
        if (this.isComplete) {
            return;
        }

        this.isComplete = true;

        lib.symbols.processor_cancel_work(
            this.processor,
            this.id,
        );
    }
}
