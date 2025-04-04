import { delay } from "@std/async";

import { lib } from "@/bindings/lib.ts";
import {
    ReadbackBuffer,
    ReadbackData,
    ReadbackFinish,
} from "./readbackBuffer.ts";

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

    async readNext(): Promise<ReadbackData | ReadbackFinish | null> {
        if (this.isComplete) {
            return null;
        }

        const data = await this.readbackBuffer.readNext();
        if (data !== null) {
            return data;
        }

        if (this.readbackBuffer.isFinished()) {
            const data = await this.readbackBuffer.readNext();
            if (data !== null) {
                return data;
            }
            const status = await this.readbackBuffer.readStatus();
            this.readbackBuffer.reset();
            this.isComplete = true;
            return status;
        }

        return null;
    }

    async *stream(): AsyncGenerator<
        ReadbackData | ReadbackFinish,
        void,
        unknown
    > {
        this.readbackBuffer.reset();

        while (!this.isComplete) {
            const data = await this.readNext();
            if (data === null) {
                if (this.isComplete) {
                    break;
                }

                await delay(10);
                continue;
            }

            yield data;

            if (data.kind === "finish") {
                break;
            }
        }
    }

    async cancel() {
        if (this.isComplete) {
            return;
        }

        const cancelled = lib.symbols.processor_cancel_work(
            this.processor,
            this.id,
        );

        if (cancelled && this.readbackBuffer.isFinished()) {
            await this.readbackBuffer.readStatus();
        }

        this.readbackBuffer.reset();
        this.isComplete = true;
    }
}
