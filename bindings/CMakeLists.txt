cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

project(DenoCppInterop)

# Fetch llama.cpp latest
# FIXME: Maybe use a vendored llama.cpp build for stability
include(FetchContent)
FetchContent_Declare(
        llama
        GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
        GIT_TAG 668750357e66bfa3d1504b65699f5a0dfe3cb7cb
)

# Required for proper builds
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Disable unused components
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama.cpp: build examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama.cpp: build tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama.cpp: build server" FORCE)

# Enable GGML_CUDA by default
# TODO: Switch this off in production
#set(GGML_CUDA ON CACHE BOOL "Enable CUDA" FORCE)
#set(GGML_STATIC ON CACHE BOOL "Static build" FORCE)

FetchContent_MakeAvailable(llama)

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

# Windows options
if(WIN32)
    set_target_properties(deno_cpp_binding PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )

    # Gather and copy all DLLs
    file(GLOB DLL_FILES
            ${CMAKE_CURRENT_BINARY_DIR}/bin/*.dll
            ${CMAKE_CURRENT_BINARY_DIR}/deno_cpp_binding.dll
    )

    add_custom_command(TARGET deno_cpp_binding POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${DLL_FILES}
            ${CMAKE_CURRENT_SOURCE_DIR}/lib
            COMMENT "Copying DLLs to lib"
    )
endif()