import { Model } from "@/bindings/bindings.ts";

export interface OAIContext {
    requestId: string;
    model: Model;
    cancellationSignal: AbortSignal;
}
