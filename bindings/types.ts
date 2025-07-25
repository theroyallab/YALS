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

export type ReadbackFinishReason =
    | "CtxExceeded"
    | "BatchDecode"
    | "StopToken"
    | "MaxNewTokens"
    | "StopString"
    | "TokenEncode"
    | "Aborted";

// MARK: C++ chunks
export interface ReadbackStreamChunk {
    kind: "data";
    text: string;
    token: number;
}

export interface ReadbackFinishChunk {
    kind: "finish";

    slotId: number;
    slotRequestId: number;
    jobIndex: number;

    promptTokens: number;
    genTokens: number;
    promptSec: number;
    genSec: number;
    totalSec: number;

    genTokensPerSec: number;
    promptTokensPerSec: number;
    finishReason: ReadbackFinishReason;
    stopToken: string;
}

export type ReadbackGenerationChunk = ReadbackStreamChunk | ReadbackFinishChunk;

// MARK: API chunks

interface BaseChunk {
    kind: string;
    taskIdx: number;
    requestId: string;
    text: string;
}

export interface StreamChunk extends BaseChunk {
    kind: "data";
}

export interface FinishChunk extends BaseChunk {
    kind: "finish";
    fullText: string;

    promptTokens: number;
    genTokens: number;

    promptSec: number;
    genSec: number;
    totalSec: number;
    promptTokensPerSec: number;
    genTokensPerSec: number;

    finishReason: ReadbackFinishReason;
    stopToken: string;

    toolCalls?: string;
}

export type GenerationChunk = StreamChunk | FinishChunk;
