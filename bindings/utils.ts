export function pointerArrayFromStrings(strings: string[]): {
    inner: BigUint64Array;
    // Return the buffer so it stays alive
    buffer: Uint8Array;
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
