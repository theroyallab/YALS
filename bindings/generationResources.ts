import { lib } from "./lib.ts";
import { ReadbackBuffer } from "./readbackBuffer.ts";

export class GenerationResources {
    private readbackBufferPtr: Deno.PointerValue;

    rawPtr: Deno.PointerValue;
    samplerPtr: Deno.PointerValue;
    readbackBuffer: ReadbackBuffer;

    constructor() {
        this.rawPtr = lib.symbols.generation_resources_make();
        if (!this.rawPtr) {
            throw new Error("Could not allocate shared resource bundle.");
        }

        const view = new Deno.UnsafePointerView(this.rawPtr);
        this.readbackBufferPtr = Deno.UnsafePointer.create(
            view.getBigUint64(0),
        );
        this.readbackBuffer = new ReadbackBuffer(this.readbackBufferPtr);

        this.samplerPtr = Deno.UnsafePointer.create(view.getBigUint64(8));
    }

    close() {
        lib.symbols.generation_resources_release(this.rawPtr);
    }
}
