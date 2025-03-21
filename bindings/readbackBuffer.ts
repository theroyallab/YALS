import { lib } from "./lib.ts";

export interface ReadbackData {
    kind: "data";
    text: string;
    token: number;
}

export interface ReadbackFinish {
    kind: "finish";
    text: string;
    [key: string]: unknown; // Additional properties from status JSON
}

/**
 * Union type of possible responses from the readback buffer
 */
export type ReadbackResponse = ReadbackData | ReadbackFinish;

/**
 * ReadbackBuffer provides an interface to read generated tokens and text
 * from the LLM generation process.
 */
export class ReadbackBuffer {
    private bufferPtr: Deno.PointerValue;
    private isFreed = false;

    /**
     * Creates a new ReadbackBuffer instance
     */
    constructor() {
        this.bufferPtr = lib.symbols.readback_create_buffer();
        if (!this.bufferPtr) {
            throw new Error("Failed to create readback buffer");
        }
    }

    rawPointer(): Deno.PointerValue {
        return this.bufferPtr;
    }

    /**
     * Checks if the buffer has finished processing
     */
    isFinished(): boolean {
        this.ensureNotFreed();
        return lib.symbols.readback_is_buffer_finished(this.bufferPtr);
    }

    /**
     * Reads the next token and text from the buffer
     * @returns A ReadbackData object or null if no more data is available
     */
    async readNext(): Promise<ReadbackData | null> {
        this.ensureNotFreed();

        const outCharPtrBuf = new Uint8Array(8); // For char**
        const outTokenBuf = new Int32Array(1); // For llama_token*

        const outCharPtr = Deno.UnsafePointer.of(outCharPtrBuf);
        const outTokenPtr = Deno.UnsafePointer.of(outTokenBuf);

        const success = await lib.symbols.readback_read_next(
            this.bufferPtr,
            outCharPtr,
            outTokenPtr,
        );

        if (!success) {
            return null;
        }

        const charPtrValue = new BigUint64Array(outCharPtrBuf.buffer)[0];

        if (charPtrValue === 0n) {
            return null;
        }

        const charPtr = Deno.UnsafePointer.create(charPtrValue);
        let text = "";

        if (charPtr) {
            const view = new Deno.UnsafePointerView(charPtr);
            let length = 0;
            while (view.getUint8(length) !== 0) {
                length++;
            }

            const buffer = new Uint8Array(length);
            for (let i = 0; i < length; i++) {
                buffer[i] = view.getUint8(i);
            }

            const decoder = new TextDecoder("utf-8");
            text = decoder.decode(buffer);
        }

        return {
            kind: "data",
            text,
            token: outTokenBuf[0],
        };
    }

    /**
     * Reads the status information from the buffer
     * @returns A ReadbackFinish object or null if status couldn't be read
     */
    async readStatus(): Promise<ReadbackFinish | null> {
        this.ensureNotFreed();

        const statusPtr = await lib.symbols.readback_read_status(
            this.bufferPtr,
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
                text: "",
                kind: "finish",
            };
        } catch (e) {
            console.error("Failed to parse status JSON:", e);
            return null;
        }
    }

    /**
     * Resets the buffer to its initial state
     */
    reset(): void {
        this.ensureNotFreed();
        lib.symbols.readback_reset(this.bufferPtr);
    }

    /**
     * Frees the underlying native buffer
     * After calling this method, the buffer can no longer be used
     */
    async free() {
        if (!this.isFreed && this.bufferPtr) {
            await lib.symbols.readback_annihilate(this.bufferPtr);
            this.isFreed = true;
            this.bufferPtr = null;
        }
    }

    /**
     * Ensures the buffer hasn't been freed before performing operations
     * @throws Error if the buffer has been freed
     */
    private ensureNotFreed() {
        if (this.isFreed) {
            throw new Error("Cannot perform operation on freed ReadbackBuffer");
        }
    }
}

/**
 * Creates a ReadbackBuffer that will be automatically freed when it goes out of scope
 * @returns A tuple with the ReadbackBuffer and a function to free it
 */
export function createReadbackBuffer(): [ReadbackBuffer, () => void] {
    const buffer = new ReadbackBuffer();
    return [buffer, async () => await buffer.free()];
}
