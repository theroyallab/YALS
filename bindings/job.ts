import { lib } from "@/bindings/lib.ts";
import { ReadbackBuffer } from "./readbackBuffer.ts";
import { GenerationChunk } from "./types.ts";

export class Job {
    isComplete = false;
    id: number;

    // Private references
    private readbackBuffer: ReadbackBuffer;
    private processor: Deno.PointerValue;

    constructor(
        id: number,
        readbackBuffer: ReadbackBuffer,
        processor: Deno.PointerValue,
    ) {
        this.id = id;
        this.readbackBuffer = readbackBuffer;
        this.processor = processor;
    }

    async *stream(): AsyncGenerator<GenerationChunk> {
        for await (const { text, token } of this.readbackBuffer.read()) {
            yield { kind: "data", text, token };
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
