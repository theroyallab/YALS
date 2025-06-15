// Subset for caching
export enum GGMLType {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    // 4 and 5 were removed (Q4_2 and Q4_3)
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
}

export enum GGMLTensorSplitMode {
    none = 0,
    layer = 1,
    row = 2,
}

export type GenerationChunk = StreamChunk | FinishChunk;

export interface StreamChunk {
    kind: "data";
    text: string;
    token: number;
}

export type ReadbackFinishReason =
    | "CtxExceeded"
    | "BatchDecode"
    | "StopToken"
    | "MaxNewTokens"
    | "StopString"
    | "TokenEncode"
    | "Aborted";

export interface FinishChunk {
    kind: "finish";
    text: string;
    slotId: number;
    requestId: number;
    jobIndex: number;

    promptTokens: number;
    genTokens: number;

    promptSec: number;
    genSec: number;
    totalSec: number;
    promptTokensPerSec: number;
    genTokensPerSec: number;

    finishReason: ReadbackFinishReason;
    stopToken: string;
}
