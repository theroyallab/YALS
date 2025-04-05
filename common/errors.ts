export class CancellationError extends Error {
    constructor(message: string = "Operation cancelled") {
        super(message);
        this.name = "CancellationError";
    }
}
