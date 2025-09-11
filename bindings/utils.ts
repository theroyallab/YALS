import { logger } from "@/common/logging.ts";

export function pointerArrayFromStrings(strings: string[]): {
    inner: BigUint64Array<ArrayBuffer>;
    // Return the buffer so it stays alive
    buffer: Uint8Array<ArrayBuffer>;
} {
    const encoder = new TextEncoder();

    // Calculate total buffer size needed including null terminators
    const encodedStrings = strings.map((str) => encoder.encode(str + "\0"));
    const totalSize = encodedStrings.reduce(
        (sum, encoded) => sum + encoded.length,
        0,
    );

    // Allocate single buffer for all strings
    const buffer = new Uint8Array(totalSize);
    const ptrArray = new BigUint64Array(strings.length);

    let offset = 0;
    strings.forEach((str, index) => {
        // Encode string with null terminator
        const encoded = encoder.encode(str + "\0");
        buffer.set(encoded, offset);

        // Store pointer to current string
        ptrArray[index] = BigInt(Deno.UnsafePointer.value(
            Deno.UnsafePointer.of(buffer.subarray(offset)),
        ));

        offset += encoded.length;
    });

    // Return both the pointer array and the buffer
    return { inner: ptrArray, buffer };
}

export function adjustCacheSize(cacheSize: number, maxSeqLen: number) {
    if (cacheSize < maxSeqLen) {
        logger.warn(
            `The given cache_size (${cacheSize}) is smaller than the ` +
                "desired context length.\n" +
                "Overriding cache_size to max_seq_len. ",
        );

        cacheSize = maxSeqLen;
    }

    const cacheRemainder = cacheSize % 256;
    if (cacheRemainder != 0) {
        const roundedCacheSize = 256 *
            Math.floor((cacheSize - cacheRemainder) / 256 + 1);
        logger.info(
            `Rounding cache size from ${cacheSize} to ${roundedCacheSize} ` +
                `tokens (multiple of 256)`,
        );
        cacheSize = roundedCacheSize;
    }

    return cacheSize;
}
