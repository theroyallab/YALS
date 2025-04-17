import { lib } from "./lib.ts";
import { GenerationResources } from "./generationResources.ts";

export interface LogitBias {
    token: number;
    bias: number;
}

export class SamplerBuilder {
    private sampler: Deno.PointerValue;
    private readonly model: Deno.PointerValue;

    constructor(
        model: Deno.PointerValue,
        resourceBundle: GenerationResources,
    ) {
        this.sampler = resourceBundle.samplerPtr;
        if (!this.sampler) {
            throw new Error("Failed to create sampler");
        }
        this.model = model;
    }

    /**
     * Adds distribution sampling with the specified seed
     * @param seed Random seed for sampling
     * @returns This builder instance for chaining
     */
    dist(seed: number): SamplerBuilder {
        this.sampler = lib.symbols.sampler_dist(this.sampler, seed);
        return this;
    }

    /**
     * Adds grammar-based sampling constraints
     * @param grammar Grammar definition as a string
     * @param root Root rule name in the grammar
     * @returns This builder instance for chaining
     */
    grammar(grammar: string, root: string): SamplerBuilder {
        const grammarPtr = new TextEncoder().encode(grammar + "\0");
        const rootPtr = new TextEncoder().encode(root + "\0");

        this.sampler = lib.symbols.sampler_grammar(
            this.sampler,
            this.model,
            grammarPtr,
            rootPtr,
        );

        return this;
    }

    /**
     * Adds llguidance sampler
     * @param grammar Grammar definition as a string
     */
    llguidance(grammar: string): SamplerBuilder {
        const grammarPtr = new TextEncoder().encode(grammar + "\0");

        this.sampler = lib.symbols.sampler_llguidance(
            this.sampler,
            this.model,
            grammarPtr,
        );

        return this;
    }

    /**
     * Configures the sampler to always choose the most likely token (greedy sampling)
     * @returns This builder instance for chaining
     */
    greedy(): SamplerBuilder {
        this.sampler = lib.symbols.sampler_greedy(this.sampler);
        return this;
    }

    /**
     * Configures the sampler for infill generation
     * @returns This builder instance for chaining
     */
    infill(): SamplerBuilder {
        this.sampler = lib.symbols.sampler_infill(this.sampler, this.model);
        return this;
    }

    /**
     * Applies token biases to influence generation probabilities
     * @param logitBias Array of token biases to apply
     * @returns This builder instance for chaining
     */
    logitBias(logitBias: LogitBias[]): SamplerBuilder {
        const nBias = logitBias.length;

        const bufferSize = nBias * 8; // 4 bytes for token (int32) + 4 bytes for bias (float)
        const buffer = new ArrayBuffer(bufferSize);
        const view = new DataView(buffer);

        logitBias.forEach((bias, index) => {
            view.setInt32(index * 8, bias.token, true);
            view.setFloat32(index * 8 + 4, bias.bias, true);
        });

        this.sampler = lib.symbols.sampler_logit_bias(
            this.sampler,
            this.model,
            nBias,
            Deno.UnsafePointer.of(buffer),
        );

        return this;
    }

    /**
     * Configures dry run sampling with sequence breakers
     * @param multiplier Dry run multiplier
     * @param base Dry run base
     * @param allowedLength Maximum allowed length
     * @param penaltyLastN Penalty context window size
     * @param sequenceBreakers Array of strings that break sequences
     * @returns This builder instance for chaining
     */
    dry(
        multiplier: number,
        base: number,
        allowedLength: number,
        penaltyLastN: number,
        sequenceBreakers: string[] = [],
    ): SamplerBuilder {
        const nullTerminatedBreakers = sequenceBreakers.map((str) =>
            str + "\0"
        );

        // Encode strings to Uint8Arrays
        const encodedBreakers = nullTerminatedBreakers.map((str) =>
            new TextEncoder().encode(str)
        );

        // Create pointers to encoded strings
        const breakerPtrs = encodedBreakers.map((encoded) =>
            Deno.UnsafePointer.of(encoded)
        );

        // Create an array to hold the pointers
        const ptrArrayBuffer = new ArrayBuffer(breakerPtrs.length * 8);
        const ptrArray = new BigUint64Array(ptrArrayBuffer);

        // Store the pointer values in the array
        breakerPtrs.forEach((ptr, index) => {
            ptrArray[index] = BigInt(Deno.UnsafePointer.value(ptr));
        });

        this.sampler = lib.symbols.sampler_dry(
            this.sampler,
            this.model,
            multiplier,
            base,
            allowedLength,
            penaltyLastN,
            Deno.UnsafePointer.of(ptrArrayBuffer),
            BigInt(sequenceBreakers.length),
        );

        return this;
    }

    /**
     * Configures minimum-p sampling
     * @param minP Minimum probability threshold
     * @param minKeep Minimum number of tokens to keep
     * @returns This builder instance for chaining
     */
    minP(minP: number, minKeep: bigint): SamplerBuilder {
        this.sampler = lib.symbols.sampler_min_p(this.sampler, minP, minKeep);
        return this;
    }

    /**
     * Configures mirostat sampling (adaptive temperature)
     * @param seed Random seed
     * @param tau Target entropy
     * @param eta Learning rate
     * @param m Order of the mirostat
     * @returns This builder instance for chaining
     */
    mirostat(
        seed: number,
        tau: number,
        eta: number,
        m: number,
    ): SamplerBuilder {
        this.sampler = lib.symbols.sampler_mirostat(
            this.sampler,
            this.model,
            seed,
            tau,
            eta,
            m,
        );
        return this;
    }

    /**
     * Configures mirostat v2 sampling (simplified adaptive temperature)
     * @param seed Random seed
     * @param tau Target entropy
     * @param eta Learning rate
     * @returns This builder instance for chaining
     */
    mirostatV2(seed: number, tau: number, eta: number): SamplerBuilder {
        this.sampler = lib.symbols.sampler_mirostat_v2(
            this.sampler,
            seed,
            tau,
            eta,
        );
        return this;
    }

    /**
     * Configures repetition penalties
     * @param penaltyLastN Number of tokens to consider for penalties
     * @param penaltyRepeat Repetition penalty
     * @param penaltyFreq Frequency penalty
     * @param penaltyPresent Presence penalty
     * @returns This builder instance for chaining
     */
    penalties(
        penaltyLastN: number,
        penaltyRepeat: number,
        penaltyFreq: number,
        penaltyPresent: number,
    ): SamplerBuilder {
        this.sampler = lib.symbols.sampler_penalties(
            this.sampler,
            penaltyLastN,
            penaltyRepeat,
            penaltyFreq,
            penaltyPresent,
        );
        return this;
    }

    /**
     * Sets the sampling temperature
     * @param temp Temperature value (higher = more random)
     * @returns This builder instance for chaining
     */
    temp(temp: number): SamplerBuilder {
        this.sampler = lib.symbols.sampler_temp(this.sampler, temp);
        return this;
    }

    /**
     * Sets extended temperature settings
     * @param temp Base temperature
     * @param dynatempRange Dynamic temperature range
     * @param dynatempExponent Dynamic temperature exponent
     * @returns This builder instance for chaining
     */
    tempExt(
        temp: number,
        dynatempRange: number,
        dynatempExponent: number,
    ): SamplerBuilder {
        this.sampler = lib.symbols.sampler_temp_ext(
            this.sampler,
            temp,
            dynatempRange,
            dynatempExponent,
        );
        return this;
    }

    /**
     * Configures top-k sampling
     * @param k Number of most likely tokens to consider
     * @returns This builder instance for chaining
     */
    topK(k: number): SamplerBuilder {
        this.sampler = lib.symbols.sampler_top_k(this.sampler, k);
        return this;
    }

    /**
     * Configures top-p (nucleus) sampling
     * @param p Cumulative probability threshold
     * @param minKeep Minimum number of tokens to keep
     * @returns This builder instance for chaining
     */
    topP(p: number, minKeep: bigint): SamplerBuilder {
        this.sampler = lib.symbols.sampler_top_p(this.sampler, p, minKeep);
        return this;
    }

    /**
     * Configures typical sampling
     * @param typicalP Typical probability threshold
     * @param minKeep Minimum number of tokens to keep
     * @returns This builder instance for chaining
     */
    typical(typicalP: number, minKeep: bigint): SamplerBuilder {
        this.sampler = lib.symbols.sampler_typical(
            this.sampler,
            typicalP,
            minKeep,
        );
        return this;
    }

    /**
     * Configures top-n-sigma sampling
     * @param nSigma Number of standard deviations to consider
     * @returns This builder instance for chaining
     */
    topNSigma(nSigma: number): SamplerBuilder {
        this.sampler = lib.symbols.sampler_top_n_sigma(this.sampler, nSigma);
        return this;
    }

    /**
     * Configures XTC (exploration time control) sampling
     * @param xtcProbability XTC probability
     * @param xtcThreshold XTC threshold
     * @param minKeep Minimum number of tokens to keep
     * @param seed Random seed
     * @returns This builder instance for chaining
     */
    xtc(
        xtcProbability: number,
        xtcThreshold: number,
        minKeep: bigint,
        seed: number,
    ): SamplerBuilder {
        this.sampler = lib.symbols.sampler_xtc(
            this.sampler,
            xtcProbability,
            xtcThreshold,
            minKeep,
            seed,
        );
        return this;
    }

    /**
     * Builds and returns the configured sampler
     * @returns Pointer to the configured sampler
     */
    build(): Deno.PointerValue {
        return this.sampler;
    }
}
