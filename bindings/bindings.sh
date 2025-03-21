#!/bin/bash

OS=$(uname -s)

# Set number of jobs for parallel build
if [ -n "$MAX_JOBS" ]; then
    JOBS=$MAX_JOBS
elif [ "$OS" = "Darwin" ]; then
    JOBS=$(sysctl -n hw.physicalcpu)
else
    JOBS=$(nproc --all)
fi

# Initialize as empty array
EXTRA_CMAKE_ARGS=()

# llama.cpp dev options
if [ -n "$LLAMACPP_REPO" ]; then
    EXTRA_CMAKE_ARGS+=("-DLLAMACPP_REPO=$LLAMACPP_REPO")
    echo "Using custom llama.cpp repo: ${LLAMACPP_REPO}"
fi

if [ -n "$LLAMACPP_COMMIT" ]; then
    EXTRA_CMAKE_ARGS+=("-DLLAMACPP_COMMIT=$LLAMACPP_COMMIT")
    echo "Using custom llama.cpp commit: ${LLAMACPP_COMMIT}"
fi

if [ "$GGML_CUDA" = "1" ]; then
    EXTRA_CMAKE_ARGS+=("-DGGML_CUDA=ON")
    echo "CUDA enabled, including in build"

    if [ -n "$CMAKE_CUDA_ARCHITECTURES" ]; then
        EXTRA_CMAKE_ARGS+=(
            "-DGGML_NATIVE=OFF" "-DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"
        )
    fi
fi

if [ "$GGML_VULKAN" = "1" ]; then
    EXTRA_CMAKE_ARGS+=("-DGGML_VULKAN=ON")
    echo "Vulkan enabled, including in build"
fi

if [ "$GGML_HIP" = "1" ]; then
    EXTRA_CMAKE_ARGS+=("-DGGML_HIP=ON")
    echo "HIP enabled, including in build"

    if [ -n "$AMDGPU_TARGETS" ]; then
        EXTRA_CMAKE_ARGS+=(
            "-DAMDGPU_TARGETS=$AMDGPU_TARGETS"
        )
    fi
fi

# Join array elements with spaces
CMAKE_ARGS="${EXTRA_CMAKE_ARGS[*]}"

cmake . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release ${CMAKE_ARGS}
cmake --build build --config Release --target c_library -j ${JOBS}

if [ "$OS" = "Darwin" ]; then
    echo "Copying .dylib files"
    cp build/bin/*.dylib ../lib
elif [ "$OS" = "Linux" ]; then
    echo "Copying .so files"
    cp build/bin/*.so ../lib
fi;
