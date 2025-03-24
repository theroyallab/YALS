import {
    ReadbackBuffer,
    ReadbackData,
    ReadbackFinish,
} from "./readbackBuffer.ts";

export class Job {
    isComplete = false;

    constructor(
        private readonly id: number,
        private readonly buffer: ReadbackBuffer,
    ) {}

    getId(): number {
        return this.id;
    }

    async readNext(): Promise<ReadbackData | ReadbackFinish | null> {
        if (this.isComplete) {
            return null;
        }

        const data = await this.buffer.readNext();
        if (data !== null) {
            return data;
        }

        if (this.buffer.isFinished()) {
            const data = await this.buffer.readNext();
            if (data !== null) {
                return data;
            }
            const status = await this.buffer.readStatus();
            this.buffer.reset();
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
        while (!this.isComplete) {
            const data = await this.readNext();
            if (data === null) {
                if (this.isComplete) {
                    break;
                }

                await new Promise((resolve) => setTimeout(resolve, 10));
                continue;
            }

            yield data;

            if (data.kind === "finish") {
                break;
            }
        }
    }
}
