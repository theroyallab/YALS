import * as Path from "@std/path";
import { delay } from "@std/async/delay";

import { config } from "@/common/config.ts";
import { ModelConfig } from "@/common/configModels.ts";
import { logGenParams, logger, logSection } from "@/common/logging.ts";
import { BaseSamplerRequest } from "@/common/sampling.ts";
import { PromptTemplate } from "@/common/templating.ts";
import { defer } from "@/common/utils.ts";
import { MaybePromise } from "@/types/utils.ts";
import { GenerationResources } from "./generationResources.ts";
import { YALSGrammar } from "./grammar.ts";
import { lib } from "./lib.ts";
import { Job } from "./job.ts";
import { SamplerBuilder } from "./samplers.ts";
import {
    FinishChunk,
    GenerationChunk,
    GGMLTensorSplitMode,
    ReadbackFinishChunk,
} from "./types.ts";
import { adjustCacheSize, pointerArrayFromStrings } from "./utils.ts";

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
    // Internal pointers
    private model: Deno.PointerValue;

    bosToken?: Token;
    eosToken?: Token;
    eotToken?: Token;
    addBosToken: boolean = true;

    constructor(model: Deno.PointerValue) {
        const bosTokenId = lib.symbols.model_vocab_bos(model);
        const eosTokenId = lib.symbols.model_vocab_eos(model);
        const eotTokenId = lib.symbols.model_vocab_eot(model);

        this.bosToken = Tokenizer.createTokenPair(model, bosTokenId);
        this.eosToken = Tokenizer.createTokenPair(model, eosTokenId);
        this.eotToken = Tokenizer.createTokenPair(model, eotTokenId);
        this.addBosToken = lib.symbols.model_vocab_add_bos(model);

        this.model = model;
    }

    static createTokenPair(model: Deno.PointerValue, tokenId: number) {
        if (tokenId == -1) {
            return undefined;
        }

        const piece = this.tokenToText(model, tokenId);
        return { id: tokenId, piece };
    }

    static tokenToText(model: Deno.PointerValue, tokenId: number) {
        const stringPtr = lib.symbols.model_vocab_token_to_string(
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
        using _ = defer(() => {
            lib.symbols.endpoint_free_tokens(tokensPtr);
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
        using _ = defer(() => {
            lib.symbols.endpoint_free_string(textPtr);
        });

        if (textPtr === null) {
            throw new Error("Detokenization failed");
        }

        // Copy to owned string
        const cString = new Deno.UnsafePointerView(textPtr);
        const text: string = cString.getCString();

        return text;
    }
}

export class Model {
    // Internal pointers
    private model: Deno.PointerValue;
    private context: Deno.PointerValue;
    private cache: Deno.PointerValue;
    private processor: Deno.PointerValue;

    // Concurrency
    private activeJobIds: Map<string, Job | undefined> = new Map();
    private closing: boolean = false;

    // Extra model info
    maxSeqLen: number;
    path: Path.ParsedPath;
    tokenizer: Tokenizer;
    promptTemplate?: PromptTemplate;

    private constructor(
        model: Deno.PointerValue,
        context: Deno.PointerValue,
        cache: Deno.PointerValue,
        processor: Deno.PointerValue,
        path: Path.ParsedPath,
        tokenizer: Tokenizer,
        maxSeqLen: number,
        promptTemplate?: PromptTemplate,
    ) {
        this.model = model;
        this.context = context;
        this.cache = cache;
        this.processor = processor;
        this.path = path;
        this.tokenizer = tokenizer;
        this.promptTemplate = promptTemplate;
        this.maxSeqLen = maxSeqLen;
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

        // Set GPU split mode
        if (params.gpu_split.length > 1) {
            if (params.gpu_split_mode === GGMLTensorSplitMode.row) {
                logger.info("Loading with row GPU split");
            } else {
                logger.info("Loading with GPU split");
            }
        } else {
            logger.info("Loading with a single GPU setup");
            params.gpu_split_mode = GGMLTensorSplitMode.none;
        }

        const tensorSplitPtr = new Float32Array(params.gpu_split);

        // MoE tensor overrides
        const tensorOverrides = new Set(params.override_tensor);
        if (params.n_cpu_moe) {
            if (params.n_cpu_moe === "all") {
                tensorOverrides.add("\\.ffn_(up|down|gate)_exps=CPU");
            } else {
                for (let i = 0; i < params.n_cpu_moe; i++) {
                    tensorOverrides.add(
                        `blk\\.${i}\\.ffn_(up|down|gate)_exps=CPU`,
                    );
                }
            }
        }

        const tensorOverrideString = tensorOverrides.size > 0
            ? new TextEncoder().encode([...tensorOverrides].join(",") + "\0")
            : null;

        const model = await lib.symbols.model_load(
            modelPathPtr,
            params.num_gpu_layers,
            params.gpu_split_mode,
            tensorSplitPtr,
            callback?.pointer ?? null,
            tensorOverrideString,
            params.mmap,
            config.developer.realtime_process_priority,
        );

        // Was the load aborted?
        if (!model) {
            return;
        }

        // Use only one thread if model is fully offloaded on GPU
        const num_model_layers = lib.symbols.model_n_layer(model);
        const model_on_gpu = params.num_gpu_layers >= num_model_layers ||
            params.num_gpu_layers === -1;

        // Don't use one thread if tensor overrides are present
        if (model_on_gpu && tensorOverrides.size == 0) {
            params.num_threads = 1;
            logger.warn("Model fully on GPU, setting num_threads to 1");
        }

        // Cast to fill the entire model
        // llamacpp uses 0 instead of -1, but this isn't user intuitive
        let maxSeqLen = params.max_seq_len === -1 ? 0 : params.max_seq_len;

        // Initialize cache size
        let cacheSize = params.cache_size ?? maxSeqLen;

        // Adjust cache size param to multiple of 256
        cacheSize = adjustCacheSize(cacheSize, maxSeqLen);

        console.log(
            `Chunk size: ${params.chunk_size}, Phys chunk size: ${params.physical_chunk_size}`,
        );

        const context = await lib.symbols.ctx_make(
            model,
            cacheSize,
            params.chunk_size,
            params.physical_chunk_size ?? params.chunk_size,
            params.num_slots,
            params.num_threads,
            params.flash_attention,
            params.rope_freq_base,
            params.enable_yarn, // Use yarn
            params.cache_mode_k,
            params.cache_mode_v,
            -1.0, //kvDefrag thresehold
            params.kv_offload,
        );

        if (!context) {
            throw new Error(
                "Model context not initialized. Read above logs for errors.",
            );
        }

        // Model-defined cache size
        cacheSize = lib.symbols.ctx_max_seq_len(context);

        const cache = await lib.symbols.memory_make(context);

        if (!cache) {
            throw new Error(
                "Model KV cache not found. Read above logs for errors.",
            );
        }

        const processor = await lib.symbols.processor_make(
            model,
            context,
            cache,
            params.num_slots,
        );

        // Adjust the maxSeqLen to be the full context if -1
        // This needs to be done after cache size is established
        if (params.max_seq_len == -1) {
            maxSeqLen = cacheSize;
        }

        const parsedModelPath = Path.parse(modelPath);
        const tokenizer = new Tokenizer(model);
        const findTemplateFunctions: MaybePromise<PromptTemplate>[] = [
            () => Model.getChatTemplate(model),
        ];

        let promptTemplate: PromptTemplate | undefined = undefined;
        if (params.prompt_template) {
            findTemplateFunctions.unshift(
                () =>
                    PromptTemplate.fromFile(
                        `templates/${params.prompt_template}`,
                    ),
            );
        }

        for (const templateFunc of findTemplateFunctions) {
            try {
                if (!promptTemplate) {
                    const result = templateFunc();
                    promptTemplate = result instanceof Promise
                        ? await result
                        : result;
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

        logger.info(
            `Using processor with a cache size of ${cacheSize}, ` +
                `max sequence length of ${maxSeqLen}, ` +
                `and ${params.num_slots} slot(s)`,
        );

        return new Model(
            model,
            context,
            cache,
            processor,
            parsedModelPath,
            tokenizer,
            maxSeqLen,
            promptTemplate,
        );
    }

    resetKVCache() {
        lib.symbols.memory_clear(this.cache);
    }

    async waitForJobs(skipWait: boolean = false) {
        // Immediately terminate all jobs if skip is true
        if (skipWait) {
            logger.warn(
                "Immediately terminating all jobs. Clients will have their requests cancelled.\n",
            );

            for (const wrappedJob of this.activeJobIds.values()) {
                if (wrappedJob) {
                    wrappedJob.cancel();
                }
            }
        }

        while (true) {
            if (this.activeJobIds.size === 0) {
                break;
            }

            await delay(10);
        }
    }

    async unload(skipWait: boolean = false) {
        // Tell all incoming jobs that the model is being closed
        this.closing = true;

        // Wait for jobs to complete
        await this.waitForJobs(skipWait);

        lib.symbols.model_free(this.model);
        lib.symbols.ctx_free(this.context);
        lib.symbols.processor_free(this.processor);
    }

    async generate(
        requestId: string,
        prompt: string,
        params: BaseSamplerRequest,
        abortSignal: AbortSignal,
        taskIdx: number = 0,
    ): Promise<FinishChunk> {
        let result: FinishChunk | undefined;

        const generator = this.generateGen(
            requestId,
            prompt,
            params,
            abortSignal,
            taskIdx,
        );

        for await (const chunk of generator) {
            if (chunk.kind === "finish") {
                result = chunk;
                break;
            }
        }

        // Fire if a finish chunk isn't found
        // This means the generation didn't complete properly
        if (!result) {
            throw new Error(
                "Generation completed without receiving finish chunk",
            );
        }

        // Static text is the full text
        // Do this here since streaming can output the fullText again
        return {
            ...result,
            text: result.fullText,
        };
    }

    handleReadbackFinish(
        requestId: string,
        finishResponse: ReadbackFinishChunk,
        fullText: string,
        taskIdx: number,
    ): FinishChunk {
        switch (finishResponse.finishReason) {
            case "CtxExceeded":
                throw new Error(
                    `Prompt exceeds max context length of ${this.maxSeqLen}`,
                );

            case "BatchDecode":
                throw new Error(
                    "Internal generation state is broken due to llama_decode error. " +
                        "Please restart the server.",
                );

            case "TokenEncode":
                throw new Error(
                    "Could not tokenize the provided prompt. " +
                        "Please make sure your prompt is formatted correctly.",
                );
        }

        const totalTime = finishResponse.promptSec + finishResponse.genSec;
        logger.info(
            `Metrics (ID: ${requestId}): ` +
                `${finishResponse.genTokens} tokens ` +
                `generated in ${totalTime.toFixed(2)} seconds ` +
                `(Prompt: ${finishResponse.promptTokens} tokens in ` +
                `${finishResponse.promptTokensPerSec.toFixed(2)} T/s, ` +
                `Generate: ${finishResponse.genTokensPerSec.toFixed(2)} T/s, ` +
                `Context: ${
                    finishResponse.promptTokens + finishResponse.genTokens
                } tokens)`,
        );

        return {
            ...finishResponse,
            text: "",
            fullText,
            taskIdx,
            requestId,
        };
    }

    async *generateGen(
        requestId: string,
        prompt: string,
        params: BaseSamplerRequest,
        abortSignal: AbortSignal,
        taskIdx: number = 0,
    ): AsyncGenerator<GenerationChunk> {
        // Get out if the model is shutting down
        if (this.closing) {
            throw new Error(
                "Model is being unloaded. Cannot process new generation requests.",
            );
        }

        const genResources = new GenerationResources();

        using _ = defer(() => {
            // Log generation params to console
            logGenParams(requestId, params);

            // Remove ID from active jobs
            this.activeJobIds.delete(requestId);

            // Mark shared generation resources for freeing
            genResources.close();
        });

        // Append the Job ID first
        this.activeJobIds.set(requestId, undefined);

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

        const samplerBuilder = new SamplerBuilder(
            this.model,
            genResources,
        );

        // Grammar
        const grammarBuilder = new YALSGrammar(samplerBuilder);
        if (params.json_schema && Object.keys(params.json_schema).length > 0) {
            grammarBuilder.jsonSchema(params.json_schema);
        }

        if (params.regex_pattern) {
            grammarBuilder.regex(params.regex_pattern);
        }

        if (params.grammar_string) {
            grammarBuilder.BNF(params.grammar_string);
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

        // Use Aphrodite's sampler position
        if (params.nsigma > 0) {
            samplerBuilder.topNSigma(params.nsigma);
        }

        samplerBuilder.topK(params.top_k);
        samplerBuilder.topP(params.top_p, 1n);
        samplerBuilder.minP(params.min_p, 1n);
        samplerBuilder.typical(params.typical, 1n);

        // TODO: Add mirostat mode 1, not really used though.
        if (params.mirostat_mode === 2) {
            samplerBuilder.mirostatV2(
                seed,
                params.mirostat_tau,
                params.mirostat_eta,
            );
        }

        if (params.temperature_last) {
            samplerBuilder.temp(params.temperature);
        }

        if (params.xtc_probability > 0) {
            samplerBuilder.xtc(
                params.xtc_probability,
                params.xtc_threshold,
                1n,
                seed,
            );
        }

        samplerBuilder.dist(seed);

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

        // Fallback to the model's preference
        // Ideally, this shouldn't be exposed, but frontends want it.
        const addBosToken = params.add_bos_token ?? this.tokenizer.addBosToken;
        const promptBosToken = addBosToken
            ? this.tokenizer.bosToken?.piece
            : "";

        if (config.logging.log_prompt) {
            logSection(
                "Prompt",
                promptBosToken + prompt,
            );
        }

        const jobId = lib.symbols.processor_submit_work(
            this.processor,
            promptPtr,
            genResources.rawPtr,
            params.max_tokens,
            params.min_tokens, // min_tokens
            this.maxSeqLen,
            seed,
            rewindPtrArray.inner,
            params.banned_strings.length,
            stopStringsPtr.inner,
            stopStrings.length,
            stopTokensPtr,
            stopTokens.length,
            addBosToken,
        );

        // Add the new job to active jobs for cancellation if needed
        const job = new Job(
            jobId,
            genResources.readbackBuffer,
            this.processor,
        );
        this.activeJobIds.set(requestId, job);

        let fullText = "";

        // Read from the read buffer
        for await (const chunk of job.stream()) {
            if (abortSignal.aborted) {
                job.cancel();
                abortSignal.throwIfAborted();
            }

            switch (chunk.kind) {
                case "data":
                    fullText += chunk.text;

                    yield {
                        ...chunk,
                        taskIdx,
                        requestId,
                    };
                    break;
                case "finish":
                    if (config.logging.log_prompt) {
                        logSection("Response", fullText);
                    }

                    yield this.handleReadbackFinish(
                        requestId,
                        chunk,
                        fullText,
                        taskIdx,
                    );
                    break;
            }
        }
    }

    static getChatTemplate(model: Deno.PointerValue) {
        const templatePtr = lib.symbols.model_chat_template(model);

        using _ = defer(() => {
            lib.symbols.endpoint_free_string(templatePtr);
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
