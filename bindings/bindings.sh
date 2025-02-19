#!/bin/bash

if [ "$GGML_CUDA" = "1" ]; then
    EXTRA_CMAKE_ARGS+="-DGGML_CUDA=ON"
    echo "CUDA enabled, including in build"
fi

cmake . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release ${EXTRA_CMAKE_ARGS}
cmake --build build --config Release --target deno_cpp_binding
