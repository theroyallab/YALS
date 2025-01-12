export function defer(callback: () => void): Disposable {
    return {
        [Symbol.dispose]: () => callback(),
    };
}

export function asyncDefer(callback: () => Promise<void>): AsyncDisposable {
    return {
        [Symbol.asyncDispose]: async () => await callback(),
    };
}
