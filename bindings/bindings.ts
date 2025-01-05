import { Mutex } from "@core/asyncutil";
import { delay } from "@std/async";
import * as Path from "@std/path";
import { ModelConfig } from "@/common/configModels.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";

import llamaSymbols from "./symbols.ts";
import { pointerArrayFromStrings } from "./utils.ts";
import { PromptTemplate } from "@/common/templating.ts";
import { logger } from "@/common/logging.ts";

// TODO: Move this somewhere else
interface LogitBias {
    token: number; // This corresponds to llama_token (int32_t)
    bias: number; // This corresponds to float
}

const ModelLoadCallback = {
    parameters: [
        "f32",
        "pointer",
    ], // float and void*
    result: "bool",
} as const;

const AbortCallback = {
    parameters: [
        "pointer",
    ], // float and void*
    result: "bool",
} as const;

// Automatically setup the lib
const lib = (() => {
    const libName = "deno_cpp_binding";

    const libDir = `${Deno.cwd()}/lib/`;
    let libPath = libDir;

    switch (Deno.build.os) {
        case "windows":
            Deno.env.set("PATH", `${Deno.env.get("PATH")};${libDir}`);
            libPath += `${libName}.dll`;
            break;
        case "linux":
            Deno.env.set(
                "LD_LIBRARY_PATH",
                `${Deno.env.get("LD_LIBRARY_PATH")}:${libDir}`,
            );
            libPath += `${libName}.so`;
            break;
        case "darwin":
            Deno.env.set(
                "DYLD_LIBRARY_PATH",
                `${Deno.env.get("DYLD_LIBRARY_PATH")}:${libDir}`,
            );
            libPath += `${libName}.dylib`;
            break;
        default:
            throw new Error(`Unsupported operating system: ${Deno.build.os}`);
    }

    return Deno.dlopen(libPath, llamaSymbols);
})();

export class SamplerBuilder {
    private sampler: Deno.PointerValue;
    private readonly model: Deno.PointerValue;

    constructor(
        model: Deno.PointerValue,
    ) {
        this.sampler = lib.symbols.MakeSampler();
        this.model = model;
    }

    distSampler(seed: number) {
        this.sampler = lib.symbols.DistSampler(this.sampler, seed);
    }

    grammarSampler(
        model: Deno.PointerValue,
        grammar: string,
        root: string,
    ) {
        const grammarPtr = new TextEncoder().encode(grammar + "\0");
        const rootPtr = new TextEncoder().encode(root + "\0");
        this.sampler = lib.symbols.GrammarSampler(
            this.sampler,
            model,
            grammarPtr,
            rootPtr,
        );
    }

    greedy() {
        this.sampler = lib.symbols.GreedySampler(this.sampler);
    }

    infillSampler(model: Deno.PointerValue) {
        this.sampler = lib.symbols.InfillSampler(this.sampler, model);
    }

    logitBiasSampler(logitBias: LogitBias[]) {
        const nBias = logitBias.length;

        // Create a buffer to hold the llama_logit_bias structures
        const bufferSize = nBias * 8; // 4 bytes for token (int32) + 4 bytes for bias (float)
        const buffer = new ArrayBuffer(bufferSize);
        const view = new DataView(buffer);

        // Fill the buffer with the logit bias data
        // only works for little endian rn
        logitBias.forEach((bias, index) => {
            view.setInt32(index * 8, bias.token, true);
            view.setFloat32(index * 8 + 4, bias.bias, true);
        });

        // Get a pointer to the buffer
        const ptr = Deno.UnsafePointer.of(buffer);

        this.sampler = lib.symbols.LogitBiasSampler(
            this.sampler,
            this.model,
            nBias,
            ptr,
        );
    }

    drySampler(
        multiplier: number,
        base: number,
        allowedLength: number,
        penaltyLastN: number,
        sequenceBreakers: string[] = [],
    ) {
        //cstring
        const nullTerminatedBreakers = sequenceBreakers.map((str) =>
            str + "\0"
        );

        //breakers
        const encodedBreakers = nullTerminatedBreakers.map((str) =>
            new TextEncoder().encode(str)
        );

        //make a pointer for each breakers e.g. char*
        const breakerPtrs = encodedBreakers.map((encoded) =>
            Deno.UnsafePointer.of(encoded)
        );

        // make a char[]* buffer
        const ptrArrayBuffer = new ArrayBuffer(breakerPtrs.length * 8);
        const ptrArray = new BigUint64Array(ptrArrayBuffer);

        // Put each pointer into an array, e.g: char[]*
        breakerPtrs.forEach((ptr, index) => {
            ptrArray[index] = BigInt(Deno.UnsafePointer.value(ptr));
        });

        this.sampler = lib.symbols.DrySampler(
            this.sampler,
            this.model,
            multiplier,
            base,
            allowedLength,
            penaltyLastN,
            ptrArrayBuffer,
            BigInt(sequenceBreakers.length),
        );
    }

    minPSampler(minP: number, minKeep: number) {
        this.sampler = lib.symbols.MinPSampler(
            this.sampler,
            minP,
            BigInt(minKeep),
        );
    }

    mirostatSampler(
        seed: number,
        tau: number,
        eta: number,
        m: number,
    ) {
        this.sampler = lib.symbols.MirostatSampler(
            this.sampler,
            this.model,
            seed,
            tau,
            eta,
            m,
        );
    }

    mirostatV2Sampler(seed: number, tau: number, eta: number) {
        this.sampler = lib.symbols.MirostatV2Sampler(
            this.sampler,
            seed,
            tau,
            eta,
        );
    }

    penaltiesSampler(
        penaltyLastN: number,
        penaltyRepeat: number,
        penaltyFreq: number,
        penaltyPresent: number,
    ) {
        this.sampler = lib.symbols.PenaltiesSampler(
            this.sampler,
            penaltyLastN,
            penaltyRepeat,
            penaltyFreq,
            penaltyPresent,
        );
    }

    tempSampler(temp: number) {
        this.sampler = lib.symbols.TempSampler(this.sampler, temp);
    }

    tempExtSampler(
        temp: number,
        dynatempRange: number,
        dynatempExponent: number,
    ) {
        this.sampler = lib.symbols.TempExtSampler(
            this.sampler,
            temp,
            dynatempRange,
            dynatempExponent,
        );
    }

    topK(num: number) {
        this.sampler = lib.symbols.TopKSampler(this.sampler, num);
    }

    topP(p: number, minKeep: number) {
        this.sampler = lib.symbols.TopPSampler(
            this.sampler,
            p,
            BigInt(minKeep),
        );
    }

    typicalSampler(typicalP: number, minKeep: number) {
        this.sampler = lib.symbols.TypicalSampler(
            this.sampler,
            typicalP,
            BigInt(minKeep),
        );
    }

    xtcSampler(
        xtcProbability: number,
        xtcThreshold: number,
        minKeep: number,
        seed: number,
    ) {
        this.sampler = lib.symbols.XtcSampler(
            this.sampler,
            xtcProbability,
            xtcThreshold,
            BigInt(minKeep),
            seed,
        );
    }

    build(): Deno.PointerValue {
        return this.sampler;
    }
}

export class ReadbackBuffer {
    public bufferPtr: Deno.PointerValue;

    constructor() {
        this.bufferPtr = lib.symbols.CreateReadbackBuffer();
    }

    private async readNext(): Promise<string | null> {
        const stringPtr = await lib.symbols.ReadbackNext(this.bufferPtr);
        if (stringPtr === null) {
            return null;
        }
        const cString = new Deno.UnsafePointerView(stringPtr);
        return cString.getCString();
    }

    isDone(): boolean {
        return lib.symbols.IsReadbackBufferDone(this.bufferPtr);
    }

    reset() {
        lib.symbols.ResetReadbackBuffer(this.bufferPtr);
    }

    async *read(): AsyncGenerator<string, void, unknown> {
        do {
            const nextString = await this.readNext();
            if (nextString === null) {
                await delay(10);
                continue;
            }
            yield nextString;
        } while (!this.isDone());
    }
}

interface Token {
    id: number;
    piece: string;
}

class Tokenizer {
    bosToken: Token;
    eosToken: Token;
    eotToken: Token;

    private constructor(bosToken: Token, eosToken: Token, eotToken: Token) {
        this.bosToken = bosToken;
        this.eosToken = eosToken;
        this.eotToken = eotToken;
    }

    static async init(model: Deno.PointerValue) {
        const bosTokenId = lib.symbols.BosToken(model);
        const eosTokenId = lib.symbols.EosToken(model);
        const eotTokenId = lib.symbols.EotToken(model);

        const bosTokenPiece = await this.tokenToText(model, bosTokenId);
        const eosTokenPiece = await this.tokenToText(model, eosTokenId);
        const eotTokenPiece = await this.tokenToText(model, eotTokenId);

        return new Tokenizer(
            { id: bosTokenId, piece: bosTokenPiece },
            { id: eosTokenId, piece: eosTokenPiece },
            { id: eotTokenId, piece: eotTokenPiece },
        );
    }

    static async tokenToText(model: Deno.PointerValue, tokenId: number) {
        const stringPtr = await lib.symbols.TokenToString(model, tokenId);

        // Empty piece
        if (stringPtr === null) {
            return "";
        }

        const cString = new Deno.UnsafePointerView(stringPtr);
        return cString.getCString();
    }
}

export class Model {
    model: Deno.PointerValue;
    context: Deno.PointerValue;
    path: Path.ParsedPath;
    tokenizer: Tokenizer;
    readbackBuffer: ReadbackBuffer;
    promptTemplate?: PromptTemplate;

    // Concurrency
    shutdown: boolean = false;
    generationLock: Mutex = new Mutex();

    private constructor(
        model: Deno.PointerValue,
        context: Deno.PointerValue,
        path: Path.ParsedPath,
        tokenizer: Tokenizer,
        promptTemplate?: PromptTemplate,
    ) {
        this.model = model;
        this.context = context;
        this.path = path;
        this.tokenizer = tokenizer;
        this.promptTemplate = promptTemplate;
        this.readbackBuffer = new ReadbackBuffer();
    }

    static async init(
        params: ModelConfig,
        progressCallback?: (progress: number) => boolean,
    ) {
        if (!params.model_name) {
            throw new Error("Model name not provided! Skipping load.");
        }

        const modelPath = Path.join(params.model_dir, params.model_name);
        const modelPathPtr = new TextEncoder().encode(modelPath + "\0");

        let callback = undefined;
        if (progressCallback) {
            callback = new Deno.UnsafeCallback(
                ModelLoadCallback,
                progressCallback,
            );
        }

        try {
            const model = await lib.symbols.LoadModel(
                modelPathPtr,
                params.num_gpu_layers as number,
                callback?.pointer ?? null,
            );

            // Was the load aborted?
            if (!model) {
                return;
            }

            // Use 2048 for chunk size for now, need more info on what to actually use
            const context = await lib.symbols.InitiateCtx(
                model,
                params.max_seq_len ?? 4096,
                2048,
                params.flash_attention,
                params.rope_freq_base,
                params.rope_freq_scale,
                GGMLType.F16,
                GGMLType.F16
            );

            const parsedModelPath = Path.parse(modelPath);
            const tokenizer = await Tokenizer.init(model);

            let promptTemplate: PromptTemplate | undefined = undefined;
            if (params.prompt_template) {
                try {
                    promptTemplate = await PromptTemplate.fromFile(
                        `templates/${params.prompt_template}`,
                    );

                    logger.info(
                        `Using template "${promptTemplate.name}" for chat completions`,
                    );
                } catch (error) {
                    if (error instanceof Error) {
                        logger.warn(
                            "Could not create a prompt template because of the following error:\n " +
                                `${error.stack}\n\n` +
                                "YALS will continue loading without the prompt template.\n" +
                                "Please proofread the template and make sure it's compatible with huggingface's jinja subset.",
                        );
                    }
                }
            }

            return new Model(
                model,
                context,
                parsedModelPath,
                tokenizer,
                promptTemplate,
            );
        } finally {
            callback?.close();
        }
    }

    resetKVCache() {
        lib.symbols.ClearContextKVCache(this.context);
    }

    async unload(skipQueue: boolean = false) {
        // Tell all jobs that the model is being unloaded
        if (skipQueue) {
            this.shutdown = true;
        }

        // Wait for jobs to complete
        using _lock = await this.generationLock.acquire();

        await lib.symbols.FreeModel(this.model);
        await lib.symbols.FreeCtx(this.context);
    }

    async generate(
        prompt: string,
        params: BaseSamplerRequest,
        abortSignal: AbortSignal,
    ) {
        let result = "";
        const generator = this.generateGen(prompt, params, abortSignal);
        for await (const chunk of generator) {
            result += chunk;
        }

        return result;
    }

    async *generateGen(
        prompt: string,
        params: BaseSamplerRequest,
        abortSignal: AbortSignal,
    ) {
        // Get out if the model is shutting down
        if (this.shutdown) {
            return;
        }

        // Acquire the mutex
        using _lock = await this.generationLock.acquire();

        // Clear generation cache
        this.resetKVCache();

        const samplerBuilder = new SamplerBuilder(this.model);
        const seed = params.seed && params.seed > 0
            ? params.seed
            : Math.floor(Math.random() * (0xFFFFFFFF + 1));

        const logitBias: LogitBias[] = [];
        if (params.logit_bias) {
            for (const [tokenId, bias] of Object.entries(params.logit_bias)) {
                logitBias.push({
                    token: parseInt(tokenId),
                    bias: bias,
                });
            }
        }

        if (params.banned_tokens) {
            const banned_tokens = params.banned_tokens as number[];

            for (const tokenId of banned_tokens) {
                logitBias.push({
                    token: tokenId,
                    bias: -100,
                });
            }
        }

        if (params.ban_eos_token) {
            const eogLogitBias: LogitBias[] = [
                { token: this.tokenizer.eosToken.id, bias: -100 },
                { token: this.tokenizer.eotToken.id, bias: -100 },
            ];

            logitBias.push(...eogLogitBias);
        }

        samplerBuilder.logitBiasSampler(logitBias);

        samplerBuilder.penaltiesSampler(
            params.penalty_range,
            params.repetition_penalty,
            params.frequency_penalty,
            params.presence_penalty,
        );

        if (params.dry_multiplier > 0) {
            samplerBuilder.drySampler(
                params.dry_multiplier,
                params.dry_base,
                params.dry_allowed_length,
                params.dry_range,
                params.dry_sequence_breakers as string[],
            );
        }

        if (!params.temperature_last) {
            samplerBuilder.tempSampler(params.temperature);
        }

        samplerBuilder.topK(params.top_k);
        samplerBuilder.topP(params.top_p, 1);
        samplerBuilder.minPSampler(params.min_p, 1);
        samplerBuilder.typicalSampler(params.typical, 1);

        if (params.xtc_probability > 0) {
            samplerBuilder.xtcSampler(
                params.xtc_probability,
                params.xtc_threshold,
                1,
                seed,
            );
        }

        if (params.temperature_last) {
            samplerBuilder.tempSampler(params.temperature);
        }

        samplerBuilder.distSampler(seed);

        const sampler = samplerBuilder.build();

        const promptPtr = new TextEncoder().encode(prompt + "\0");

        // Use the helper function for both arrays
        const rewindPtrArray = pointerArrayFromStrings(params.banned_strings);
        const stopPtrArray = pointerArrayFromStrings(params.stop);

        // Callback to abort the current generation
        const abortCallback: () => boolean = () => {
            return abortSignal.aborted || this.shutdown;
        };

        const abortCallbackPointer = new Deno.UnsafeCallback(
            AbortCallback,
            abortCallback,
        ).pointer;

        lib.symbols.InferToReadbackBuffer(
            this.model,
            sampler,
            this.context,
            this.readbackBuffer.bufferPtr,
            promptPtr,
            params.max_tokens ?? 150,
            params.add_bos_token,
            !params.skip_special_tokens,
            abortCallbackPointer,
            rewindPtrArray,
            params.banned_strings.length,
            stopPtrArray,
            params.stop.length,
        );

        // Read from the read buffer
        for await (const chunk of this.readbackBuffer.read()) {
            yield chunk;
        }
        this.readbackBuffer.reset();

        console.log("Finished");
    }
}

export enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4 and 5 were removed (Q4_2 and Q4_3)
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    // 31-33 were removed (Q4_0_4_4, Q4_0_4_8, Q4_0_8_8)
    TQ1_0 = 34,
    TQ2_0 = 35,
    // 36-38 were removed (IQ4_NL_4_4, IQ4_NL_4_8, IQ4_NL_8_8)
    COUNT = 39
}