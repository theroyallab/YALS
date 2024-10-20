cmake_minimum_required(VERSION 3.29.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

project(DenoCppInterop)

#add_compile_definitions(GGML_USE_CUDA)
#add_compile_definitions(GGML_USE_CUBLAS)

include(FetchContent)
FetchContent_Declare(
        llama
        GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
        GIT_TAG master
)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Disable unused components
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama.cpp: build examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama.cpp: build tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama.cpp: build server" FORCE)

#GGML_CUDA=1
set(GGML_CUDA ON CACHE BOOL "Enable CUDA" FORCE)

FetchContent_MakeAvailable(llama)

## THE ORDER IS IMPORTANT! FETCH CONTENT BUILDS WITH THE SETTINGS DEFINED _ABOVE_ IT!
if(STDCLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
    set(CMAKE_C_COMPILER /usr/bin/clang)
    set(CMAKE_CXX_COMPILER /usr/bin/clang++)
endif()

add_library(deno_cpp_binding SHARED
        binding.cpp
)

target_include_directories(deno_cpp_binding
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${llama_SOURCE_DIR}
)

target_link_libraries(deno_cpp_binding PRIVATE llama)

set_target_properties(deno_cpp_binding PROPERTIES
        OUTPUT_NAME "deno_cpp_binding"
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)

if(WIN32)
    set_target_properties(deno_cpp_binding PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )

    add_custom_command(TARGET deno_cpp_binding POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/bin/*.dll
            ${CMAKE_SOURCE_DIR}/lib/
    )
endif()