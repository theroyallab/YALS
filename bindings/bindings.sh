#!/bin/bash

# Set number of jobs for parallel build
if [ -n "$MAX_JOBS" ]; then
    JOBS=$MAX_JOBS
else
    JOBS=$(nproc --all)
fi

# Initialize as empty array
EXTRA_CMAKE_ARGS=()

if [ "$GGML_CUDA" = "1" ]; then
    EXTRA_CMAKE_ARGS+=("-DGGML_CUDA=ON")
    echo "CUDA enabled, including in build"

    if [ -n "$CMAKE_CUDA_ARCHITECTURES" ]; then
        EXTRA_CMAKE_ARGS+=(
            "-DGGML_NATIVE=OFF" "-DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"
        )
    fi
fi

# Join array elements with spaces
CMAKE_ARGS="${EXTRA_CMAKE_ARGS[*]}"

cmake . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release ${CMAKE_ARGS}
cmake --build build --config Release --target deno_cpp_binding -j ${JOBS}

OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    echo "Copying .dylib files"
    cp build/bin/*.dylib ../lib
elif [ "$OS" = "Linux" ]; then
    echo "Copying .so files"
    cp build/bin/*.so ../lib
fi;
