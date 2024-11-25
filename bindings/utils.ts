export function pointerArrayFromStrings(strings: string[]): BigUint64Array {
    const encoder = new TextEncoder();

    // Calculate total buffer size needed including null terminators
    const totalSize = strings.reduce((sum, str) => sum + str.length + 1, 0);

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

    return ptrArray;
}
