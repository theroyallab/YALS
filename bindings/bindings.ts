import { delay } from "@std/async";
import llamaSymbols from "./symbols.ts";
import type { BaseSamplerRequest } from "@/api/OAI/types/completions.ts";

// TODO: Move this somewhere else
interface LogitBias {
    token: number; // This corresponds to llama_token (int32_t)
    bias: number; // This corresponds to float
}

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
            Deno.UnsafePointer.of(grammarPtr),
            Deno.UnsafePointer.of(rootPtr),
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
            BigInt(nBias),
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
            BigInt(allowedLength),
            BigInt(penaltyLastN),
            Deno.UnsafePointer.of(ptrArrayBuffer),
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
        nVocab: number,
        seed: number,
        tau: number,
        eta: number,
        m: number,
    ) {
        this.sampler = lib.symbols.MirostatSampler(
            this.sampler,
            nVocab,
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
        nVocab: number,
        eosToken: number,
        nlToken: number,
        penaltyLastN: number,
        penaltyRepeat: number,
        penaltyFreq: number,
        penaltyPresent: number,
        penalizeNl: boolean,
        ignoreEos: boolean,
    ) {
        this.sampler = lib.symbols.PenaltiesSampler(
            this.sampler,
            nVocab,
            eosToken,
            nlToken,
            penaltyLastN,
            penaltyRepeat,
            penaltyFreq,
            penaltyPresent,
            penalizeNl,
            ignoreEos,
        );
    }

    softmaxSampler() {
        this.sampler = lib.symbols.SoftmaxSampler(this.sampler);
    }

    tailFreeSampler(z: number, minKeep: number) {
        this.sampler = lib.symbols.TailFreeSampler(
            this.sampler,
            z,
            BigInt(minKeep),
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

    private isDone(): boolean {
        return lib.symbols.IsReadbackBufferDone(this.bufferPtr);
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

export class Model {
    model: Deno.PointerValue;
    context: Deno.PointerValue;

    private constructor(model: Deno.PointerValue, context: Deno.PointerValue) {
        this.model = model;
        this.context = context;
    }

    static async init(modelPath: string, gpuLayers: number) {
        const modelPathPtr = new TextEncoder().encode(modelPath + "\0");
        const model = await lib.symbols.LoadModel(
            Deno.UnsafePointer.of(modelPathPtr),
            gpuLayers,
        );

        const context = await lib.symbols.InitiateCtx(
            model,
            8192,
            1,
        );

        return new Model(model, context);
    }

    async unload() {
        await lib.symbols.FreeModel(this.model);
        await lib.symbols.FreeCtx(this.context);
    }

    async generate(
        prompt: string,
        params: BaseSamplerRequest,
    ): Promise<string> {
        const samplerBuilder = new SamplerBuilder(this.model);
        const seed = params.seed ??
            Math.floor(Math.random() * (0xFFFFFFFF + 1));

        if (!params.temperature_last) {
            samplerBuilder.tempSampler(params.temperature);
        }

        samplerBuilder.topK(params.top_k);
        samplerBuilder.topP(params.top_p, 1);
        samplerBuilder.minPSampler(params.min_p, 1);
        samplerBuilder.typicalSampler(params.typical, 1);

        // TODO: Add guards
        if (params.xtc_probability > 0) {
            samplerBuilder.xtcSampler(
                params.xtc_probability,
                params.xtc_threshold,
                1,
                seed,
            );
        }

        if (params.dry_multiplier > 0) {
            samplerBuilder.drySampler(
                params.dry_multiplier,
                params.dry_base,
                params.dry_allowed_length,
                params.dry_range,
                params.dry_sequence_breakers as string[],
            );
        }

        if (params.temperature_last) {
            samplerBuilder.tempSampler(params.temperature);
        }

        samplerBuilder.distSampler(seed);
        const sampler = samplerBuilder.build();
        const promptPtr = new TextEncoder().encode(prompt + "\0");

        const readbackBuffer = new ReadbackBuffer();

        lib.symbols.InferToReadbackBuffer(
            this.model,
            sampler,
            this.context,
            readbackBuffer.bufferPtr,
            Deno.UnsafePointer.of(promptPtr),
            params.max_tokens ?? 150,
        );

        // Read from the read buffer
        let result = "";
        for await (const nextString of readbackBuffer.read()) {
            result += nextString;
        }

        return result;
    }
}
