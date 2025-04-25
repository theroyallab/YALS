import { HTTPException } from "hono/http-exception";

export class CancellationError extends Error {
    constructor(message: string = "Operation cancelled") {
        super(message);
        this.name = "CancellationError";
    }
}

export class ModelNotLoadedError extends HTTPException {
    constructor(message: string = "A model is not loaded.") {
        super(503, { message: message });
        this.name = "ModelNotLoadedError";
    }
}
