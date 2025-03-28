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

export type GenerationChunk = StreamChunk | FinishChunk;

export interface StreamChunk {
    kind: "data";
    text: string;
    token: number;
}

export interface FinishChunk {
    kind: "finish";
    text: string;
    promptTokens: number;
    genTokens: number;
    finishReason: string;
    stopToken?: string;
}
