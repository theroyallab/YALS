import { delay } from "@std/async/delay";

import { logger } from "@/common/logging.ts";
import { lib } from "./lib.ts";
import { FinishChunk } from "@/bindings/types.ts";

/**
 * ReadbackBuffer provides an interface to read generated tokens and text
 * from the LLM generation process.
 */
export class ReadbackBuffer {
    private rawPtr: Deno.PointerValue;
    constructor(readbackPtr: Deno.PointerValue) {
        this.rawPtr = readbackPtr;
    }

    async *read() {
        while (!lib.symbols.readback_is_buffer_finished(this.rawPtr)) {
            const charBuf = new Uint8Array(8);
            const tokenBuf = new Int32Array(1);

            if (
                !await lib.symbols.readback_read_next(
                    this.rawPtr,
                    Deno.UnsafePointer.of(charBuf),
                    Deno.UnsafePointer.of(tokenBuf),
                )
            ) {
                await delay(2);
                continue;
            }

            const ptrVal = new BigUint64Array(charBuf.buffer)[0];
            if (ptrVal === 0n) continue;

            const charPtr = Deno.UnsafePointer.create(ptrVal);
            if (!charPtr) continue;

            yield {
                text: new Deno.UnsafePointerView(charPtr).getCString(),
                token: tokenBuf[0],
            };
        }
    }

    /**
     * Reads the status information from the buffer
     * @returns A ReadbackFinish object or null if status couldn't be read
     */
    async readStatus(): Promise<FinishChunk | null> {
        const statusPtr = await lib.symbols.readback_read_status(
            this.rawPtr,
        );
        if (!statusPtr) {
            return null;
        }

        const view = new Deno.UnsafePointerView(statusPtr);
        const statusStr = view.getCString();

        try {
            const status = JSON.parse(statusStr);
            return {
                ...status,
                kind: "finish",
                text: "",
            };
        } catch (e) {
            logger.error("Failed to parse status JSON:", e);
            return null;
        }
    }
}
