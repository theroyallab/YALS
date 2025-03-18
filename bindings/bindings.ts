import { Mutex } from "@core/asyncutil";
import { delay } from "@std/async";
import * as Path from "@std/path";
import { ModelConfig } from "@/common/configModels.ts";
import { logGenParams, logger, logPrompt } from "@/common/logging.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";
import { PromptTemplate } from "@/common/templating.ts";
import { asyncDefer, defer } from "@/common/utils.ts";

import llamaSymbols from "./symbols.ts";
import { pointerArrayFromStrings } from "./utils.ts";
import {
    BindingFinishReason,
    BindingFinishResponse,
    BindingStreamResponse,
    FinishChunk,
    GenerationChunk,
} from "./types.ts";

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
            libPath += `${libName}.so`;
            break;
        case "darwin":
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

    topNSigma(nSigma: number) {
        this.sampler = lib.symbols.TopNSigmaSampler(
            this.sampler,
            nSigma,
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

    private async readNext(): Promise<BindingStreamResponse | null> {
        // Create a buffer for the char pointer and token
        const outCharPtr = new Uint8Array(8);
        const outTokenPtr = new Int32Array(1);

        const success = await lib.symbols.ReadbackNext(
            this.bufferPtr,
            Deno.UnsafePointer.of(outCharPtr),
            Deno.UnsafePointer.of(outTokenPtr),
        );

        if (!success) return null;

        // Read the pointer value from outCharPtr
        let char = "";
        const charPtr = new BigUint64Array(outCharPtr.buffer)[0];

        // Convert to owned
        const ownedCharPtr = Deno.UnsafePointer.create(charPtr);
        if (ownedCharPtr) {
            char = new Deno.UnsafePointerView(ownedCharPtr).getCString();
        }

        return {
            kind: "data",
            text: char,
            token: outTokenPtr[0],
        };
    }

    public async readJsonStatus(): Promise<BindingFinishResponse | null> {
        const stringPtr = await lib.symbols.ReadbackJsonStatus(this.bufferPtr);
        if (stringPtr === null) {
            return null;
        }

        const cString = new Deno.UnsafePointerView(stringPtr);
        const jsonString = cString.getCString();

        try {
            return {
                ...JSON.parse(jsonString),
                text: "",
                kind: "finish",
            };
        } catch (e) {
            console.error("Failed to parse JSON:", e);
            return null;
        }
    }

    isDone(): boolean {
        return lib.symbols.IsReadbackBufferDone(this.bufferPtr);
    }

    reset() {
        lib.symbols.ResetReadbackBuffer(this.bufferPtr);
    }

    async *read(): AsyncGenerator<BindingStreamResponse, void, unknown> {
        while (true) {
            const next = await this.readNext();
            if (next !== null) {
                yield next;
                continue;
            }

            // Only check the done status when data has been fetched
            if (this.isDone()) {
                break;
            }

            await delay(10);
        }
    }
}

interface Token {
    id: number;
    piece: string;
}

class Tokenizer {
    bosToken?: Token;
    eosToken?: Token;
    eotToken?: Token;

    private constructor(bosToken?: Token, eosToken?: Token, eotToken?: Token) {
        this.bosToken = bosToken;
        this.eosToken = eosToken;
        this.eotToken = eotToken;
    }

    static async init(model: Deno.PointerValue) {
        const bosTokenId = lib.symbols.BosToken(model);
        const eosTokenId = lib.symbols.EosToken(model);
        const eotTokenId = lib.symbols.EotToken(model);

        const bosToken = await this.createTokenPair(model, bosTokenId);
        const eosToken = await this.createTokenPair(model, eosTokenId);
        const eotToken = await this.createTokenPair(model, eotTokenId);

        return new Tokenizer(bosToken, eosToken, eotToken);
    }

    static async createTokenPair(model: Deno.PointerValue, tokenId: number) {
        if (tokenId == -1) {
            return undefined;
        }

        const piece = await this.tokenToText(model, tokenId);
        return { id: tokenId, piece };
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

    // Extra model info
    maxSeqLen: number;

    private constructor(
        model: Deno.PointerValue,
        context: Deno.PointerValue,
        path: Path.ParsedPath,
        tokenizer: Tokenizer,
        maxSeqLen: number,
        promptTemplate?: PromptTemplate,
    ) {
        this.model = model;
        this.context = context;
        this.path = path;
        this.tokenizer = tokenizer;
        this.promptTemplate = promptTemplate;
        this.maxSeqLen = maxSeqLen;
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

        // Always close progress callback
        using _ = defer(() => {
            callback?.close();
        });

        const tensorSplitPtr = new Float32Array(params.gpu_split);
        const model = await lib.symbols.LoadModel(
            modelPathPtr,
            params.num_gpu_layers,
            tensorSplitPtr,
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
            params.num_gpu_layers,
            512,
            params.flash_attention,
            params.rope_freq_base,
            params.enable_yarn, // Use yarn
            params.cache_mode_k,
            params.cache_mode_v,
            -1.0, //kvDefrag thresehold
        );

        if (context === null) {
            throw new Error(
                "Model context not initialized. Read above logs for errors.",
            );
        }

        const maxSeqLen = lib.symbols.MaxSeqLen(context);

        const parsedModelPath = Path.parse(modelPath);
        const tokenizer = await Tokenizer.init(model);
        const findTemplateFunctions = [
            Model.getChatTemplate(model),
        ];

        let promptTemplate: PromptTemplate | undefined = undefined;
        if (params.prompt_template) {
            findTemplateFunctions.unshift(
                PromptTemplate.fromFile(`templates/${params.prompt_template}`),
            );
        }

        for (const templateFunc of findTemplateFunctions) {
            try {
                if (!promptTemplate) {
                    promptTemplate = await templateFunc;
                }
            } catch (error) {
                if (error instanceof Error) {
                    logger.error(error.stack);
                    logger.warn("Attempting next template method.");
                }
            }
        }

        if (promptTemplate) {
            logger.info(`Prompt template:`);
            console.log(promptTemplate.rawTemplate);
            logger.info(
                `Using template "${promptTemplate.name}" for chat completions`,
            );
        } else {
            logger.warn(
                "Could not create a prompt template because of the above errors.\n" +
                    "Chat completions are disabled.\n" +
                    "YALS will continue loading without the prompt template.\n" +
                    "Please proofread the template and make sure it's compatible " +
                    "with huggingface's jinja subset.",
            );
        }

        return new Model(
            model,
            context,
            parsedModelPath,
            tokenizer,
            maxSeqLen,
            promptTemplate,
        );
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
    ): Promise<FinishChunk> {
        let result: FinishChunk | undefined;
        const textParts: string[] = [];

        const generator = this.generateGen(prompt, params, abortSignal);
        for await (const chunk of generator) {
            if (chunk.kind === "finish") {
                result = chunk;
            } else {
                textParts.push(chunk.text);
            }
        }

        // This shouldn't fire
        if (!result) {
            throw new Error(
                "Generation completed without receiving finish chunk",
            );
        }

        return {
            ...result,
            text: textParts.join(""),
        };
    }

    async *generateGen(
        prompt: string,
        params: BaseSamplerRequest,
        abortSignal: AbortSignal,
    ): AsyncGenerator<GenerationChunk> {
        // Cleanup operations
        using _ = defer(() => {
            // Log generation params to console
            logGenParams(params);

            this.readbackBuffer.reset();
        });

        // Get out if the model is shutting down
        if (this.shutdown) {
            return;
        }

        // Acquire the mutex
        using _lock = await this.generationLock.acquire();

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
                this.tokenizer.eosToken,
                this.tokenizer.eotToken,
            ]
                .filter((token) => token !== undefined)
                .map((token) => ({
                    token: token.id,
                    bias: -100,
                }));

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

        // TODO: Actively being changed
        // Use Aphrodite's sampler position
        if (params.nsigma > 0) {
            samplerBuilder.topNSigma(params.nsigma);
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

        // These are numbers and strings, TS doesn't understand for some reason
        const stopTokens = params.stop.filter((e) =>
            typeof e === "number"
        ) as number[];
        const stopStrings = params.stop.filter((e) =>
            typeof e === "string"
        ) as string[];

        // Use the helper function for both arrays
        const rewindPtrArray = pointerArrayFromStrings(params.banned_strings);
        const stopTokensPtr = new Uint32Array(stopTokens);
        const stopStringsPtr = pointerArrayFromStrings(stopStrings);

        const promptBosToken = params.add_bos_token
            ? this.tokenizer.bosToken?.piece
            : "";

        logPrompt(
            promptBosToken + prompt,
        );

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
            seed,
            rewindPtrArray.inner,
            params.banned_strings.length,
            stopStringsPtr.inner,
            stopStrings.length,
            stopTokensPtr,
            stopTokens.length,
        );

        // Read from the read buffer
        for await (const chunk of this.readbackBuffer.read()) {
            yield chunk;
        }

        const finishResponse = await this.readbackBuffer.readJsonStatus();
        if (finishResponse) {
            switch (finishResponse.finishReason) {
                case BindingFinishReason.CtxExceeded:
                    throw new Error(
                        `Prompt exceeds max context length of ${this.maxSeqLen}`,
                    );

                case BindingFinishReason.BatchDecode:
                    throw new Error(
                        "Internal generation state is broken due to llama_decode error. " +
                            "Please restart the server.",
                    );

                case BindingFinishReason.TokenEncode:
                    throw new Error(
                        "Could not tokenize the provided prompt. " +
                            "Please make sure your prompt is formatted correctly.",
                    );
            }

            const totalTime = finishResponse.promptSec + finishResponse.genSec;
            logger.info(
                `Metrics: ` +
                    `${finishResponse.genTokens} tokens ` +
                    `generated in ${totalTime.toFixed(2)} seconds ` +
                    `(Prompt: ${finishResponse.promptTokens} tokens in ` +
                    `${finishResponse.promptTokensPerSec.toFixed(2)} T/s, ` +
                    `Generate: ${
                        finishResponse.genTokensPerSec.toFixed(2)
                    } T/s, ` +
                    `Context: ${finishResponse.promptTokens} tokens)`,
            );

            const finishReason =
                finishResponse.finishReason == BindingFinishReason.MaxNewTokens
                    ? "length"
                    : "stop";

            yield {
                ...finishResponse,
                finishReason,
            };
        }
    }

    async tokenize(
        text: string,
        addSpecial: boolean = true,
        parseSpecial: boolean = true,
    ) {
        const textPtr = new TextEncoder().encode(text + "\0");

        const tokensPtr = await lib.symbols.EndpointTokenize(
            this.model,
            textPtr,
            addSpecial,
            parseSpecial,
        );

        // Always free the original pointer
        await using _ = asyncDefer(async () => {
            await lib.symbols.EndpointFreeTokens(tokensPtr);
        });

        if (tokensPtr === null) {
            throw new Error("Tokenization failed");
        }

        // The first 4 bytes contain the length of the array
        const ptrView = new Deno.UnsafePointerView(tokensPtr);
        const length = new Int32Array(ptrView.getArrayBuffer(4))[0];

        // Copy the actual tokens (starting after the length prefix)
        const dataPtr = Deno.UnsafePointer.create(
            Deno.UnsafePointer.value(tokensPtr) + 4n,
        )!;
        const tokenData = new Int32Array(
            new Deno.UnsafePointerView(dataPtr).getArrayBuffer(length * 4),
        );

        // Create owned copy
        const ownedTokens = new Int32Array(tokenData);

        return [...ownedTokens];
    }

    async detokenize(
        tokens: number[],
        maxTextSize: number = 4096,
        addSpecial: boolean = true,
        parseSpecial: boolean = true,
    ) {
        // Create a pointer to the tokens data
        const tokensArray = new Int32Array(tokens);
        const tokensPtr = Deno.UnsafePointer.of(tokensArray.buffer);

        // Get raw pointer from C++
        const textPtr = await lib.symbols.EndpointDetokenize(
            this.model,
            tokensPtr,
            BigInt(tokens.length),
            BigInt(maxTextSize),
            addSpecial,
            parseSpecial,
        );

        // Always free the original pointer
        await using _ = asyncDefer(async () => {
            await lib.symbols.EndpointFreeString(textPtr);
        });

        if (textPtr === null) {
            throw new Error("Detokenization failed");
        }

        // Copy to owned string
        const cString = new Deno.UnsafePointerView(textPtr);
        const text: string = cString.getCString();

        return text;
    }

    static async getChatTemplate(model: Deno.PointerValue) {
        const templatePtr = await lib.symbols.GetModelChatTemplate(model);

        await using _ = asyncDefer(async () => {
            await lib.symbols.EndpointFreeString(templatePtr);
        });

        if (templatePtr === null) {
            throw new Error("No chat template found in model");
        }

        // Copy to owned string
        const cString = new Deno.UnsafePointerView(templatePtr);
        const template: string = cString.getCString();

        return new PromptTemplate("from_gguf", template);
    }
}
