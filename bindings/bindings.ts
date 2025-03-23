import { Mutex } from "@core/asyncutil";
import * as Path from "@std/path";
import { ModelConfig } from "@/common/configModels.ts";
import { logGenParams, logger, logPrompt } from "@/common/logging.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";
import { PromptTemplate } from "@/common/templating.ts";
import { asyncDefer, defer } from "@/common/utils.ts";

import { lib } from "./lib.ts";
import { SamplerBuilder } from "./samplers.ts";
import { Job } from "./job.ts";
import {
    ReadbackBuffer,
    ReadbackFinish,
    ReadbackFinishReason,
} from "./readbackBuffer.ts";

import { pointerArrayFromStrings } from "./utils.ts";
import { FinishChunk, GenerationChunk } from "./types.ts";

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
        const bosTokenId = lib.symbols.model_vocab_bos(model);
        const eosTokenId = lib.symbols.model_vocab_eos(model);
        const eotTokenId = lib.symbols.model_vocab_eot(model);

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
        const stringPtr = await lib.symbols.model_vocab_token_to_string(
            model,
            tokenId,
        );

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
    processor: Deno.PointerValue;
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
        processor: Deno.PointerValue,
        path: Path.ParsedPath,
        tokenizer: Tokenizer,
        maxSeqLen: number,
        promptTemplate?: PromptTemplate,
    ) {
        this.model = model;
        this.context = context;
        this.processor = processor;
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
        const model = await lib.symbols.model_load(
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
        const context = await lib.symbols.ctx_make(
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

        if (!context) {
            throw new Error(
                "Model context not initialized. Read above logs for errors.",
            );
        }

        // Only create a processor with 1 slot for now
        // TODO: Make slots configurable for cont. batching
        const processor = await lib.symbols.processor_make(model, context, 1);

        const maxSeqLen = lib.symbols.ctx_max_seq_len(context);

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
            processor,
            parsedModelPath,
            tokenizer,
            maxSeqLen,
            promptTemplate,
        );
    }

    resetKVCache() {
        lib.symbols.ctx_clear_kv(this.context);
    }

    async cancelJob(job: Job) {
        if (job.isComplete) {
            return;
        }

        const cancelled = lib.symbols.processor_cancel_work(
            this.processor,
            job.getId(),
        );
        if (cancelled && this.readbackBuffer.isFinished()) {
            await this.readbackBuffer.readStatus();
        }

        this.readbackBuffer.reset();
        job.isComplete = true;
    }

    async unload(skipQueue: boolean = false) {
        // Tell all jobs that the model is being unloaded
        if (skipQueue) {
            this.shutdown = true;
        }

        // Wait for jobs to complete
        using _lock = await this.generationLock.acquire();

        await lib.symbols.model_free(this.model);
        await lib.symbols.ctx_free(this.context);
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

    handleReadbackFinish(finishResponse: ReadbackFinish): FinishChunk {
        switch (finishResponse.finishReason) {
            case ReadbackFinishReason.CtxExceeded:
                throw new Error(
                    `Prompt exceeds max context length of ${this.maxSeqLen}`,
                );

            case ReadbackFinishReason.BatchDecode:
                throw new Error(
                    "Internal generation state is broken due to llama_decode error. " +
                        "Please restart the server.",
                );

            case ReadbackFinishReason.TokenEncode:
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
                `Generate: ${finishResponse.genTokensPerSec.toFixed(2)} T/s, ` +
                `Context: ${finishResponse.promptTokens} tokens)`,
        );

        const finishReason =
            finishResponse.finishReason == ReadbackFinishReason.MaxNewTokens
                ? "length"
                : "stop";

        return {
            ...finishResponse,
            finishReason,
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

        samplerBuilder.logitBias(logitBias);

        samplerBuilder.penalties(
            params.penalty_range,
            params.repetition_penalty,
            params.frequency_penalty,
            params.presence_penalty,
        );

        if (params.dry_multiplier > 0) {
            samplerBuilder.dry(
                params.dry_multiplier,
                params.dry_base,
                params.dry_allowed_length,
                params.dry_range,
                params.dry_sequence_breakers as string[],
            );
        }

        if (!params.temperature_last) {
            samplerBuilder.temp(params.temperature);
        }

        // TODO: Actively being changed
        // Use Aphrodite's sampler position
        if (params.nsigma > 0) {
            samplerBuilder.topNSigma(params.nsigma);
        }

        samplerBuilder.topK(params.top_k);
        samplerBuilder.topP(params.top_p, 1n);
        samplerBuilder.minP(params.min_p, 1n);
        samplerBuilder.typical(params.typical, 1n);

        if (params.xtc_probability > 0) {
            samplerBuilder.xtc(
                params.xtc_probability,
                params.xtc_threshold,
                1n,
                seed,
            );
        }

        if (params.temperature_last) {
            samplerBuilder.temp(params.temperature);
        }

        samplerBuilder.dist(seed);

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
        const stopTokensPtr = new Int32Array(stopTokens);
        const stopStringsPtr = pointerArrayFromStrings(stopStrings);

        const promptBosToken = params.add_bos_token
            ? this.tokenizer.bosToken?.piece
            : "";

        logPrompt(
            promptBosToken + prompt,
        );

        //TODO::@Z
        // const larkGrammar = `
        // // Define the start rule
        // start: json_string
        // // The exact JSON string with fixed format
        // json_string: "{\\n \\"action\\" : [\\"" ACTION_CONTENT "\\"],\\n \\"mood\\" : \\"" EMOTION "\\",\\n \\"magazine capacity\\" : \\"" CAPACITY_CONTENT "\\"\\n}"
        // // Content restrictions
        // ACTION_CONTENT: /[a-zA-Z0-9 ,]{1,15}/
        // CAPACITY_CONTENT: /[0-9]+( rounds| bullets| shots)?/
        // EMOTION: "happy" | "sad" | "angry" | "excited" | "bored" | "anxious" | "calm" | "confused"
        //  | "curious" | "depressed" | "ecstatic" | "fearful" | "grateful" | "hopeful"
        //  | "irritated" | "jealous" | "peaceful" | "proud" | "surprised" | "tired"
        // `;
    

        // // Convert the string to a null-terminated buffer
        // const grammarBuffer = new TextEncoder().encode(larkGrammar + "\0");

        const jobId = await lib.symbols.processor_submit_work(
            this.processor,
            promptPtr,
            sampler,
            this.readbackBuffer.rawPointer(),
            params.max_tokens ?? 150,
            0, // min_tokens
            seed,
            rewindPtrArray.inner,
            params.banned_strings.length,
            stopStringsPtr.inner,
            stopStrings.length,
            stopTokensPtr,
            stopTokens.length,
            null
        );

        const job = new Job(jobId, this.readbackBuffer);

        // Read from the read buffer
        for await (const chunk of job.stream()) {
            if (abortSignal.aborted) {
                await this.cancelJob(job);
                break;
            }

            switch (chunk.kind) {
                case "data":
                    yield chunk;
                    break;
                case "finish":
                    yield this.handleReadbackFinish(chunk);
                    break;
            }
        }
    }

    async tokenize(
        text: string,
        addSpecial: boolean = true,
        parseSpecial: boolean = true,
    ) {
        const textPtr = new TextEncoder().encode(text + "\0");

        const tokensPtr = await lib.symbols.endpoint_tokenize(
            this.model,
            textPtr,
            addSpecial,
            parseSpecial,
        );

        // Always free the original pointer
        await using _ = asyncDefer(async () => {
            await lib.symbols.endpoint_free_tokens(tokensPtr);
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
        const textPtr = await lib.symbols.endpoint_detokenize(
            this.model,
            tokensPtr,
            tokens.length,
            maxTextSize,
            addSpecial,
            parseSpecial,
        );

        // Always free the original pointer
        await using _ = asyncDefer(async () => {
            await lib.symbols.endpoint_free_string(textPtr);
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
        const templatePtr = lib.symbols.model_chat_template(model);

        await using _ = asyncDefer(async () => {
            await lib.symbols.endpoint_free_string(templatePtr);
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
