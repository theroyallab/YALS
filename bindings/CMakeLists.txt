cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(DenoCppInterop)

# Set RPath for Apple and Unix systems
if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
elseif (UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
endif()

if (DEFINED LLAMACPP_REPO OR DEFINED LLAMACPP_COMMIT)
    message(STATUS "Using a custom commit or repo for llama.cpp. Build might not work as expected. Here be dragons!")
endif()

set(LLAMACPP_REPO "https://github.com/ggerganov/llama.cpp.git" CACHE STRING "llama.cpp repository URL")
message(STATUS "Using llama.cpp repo ${LLAMACPP_REPO}")

# Stable llama.cpp commit for bindings
set(LLAMACPP_COMMIT "84d547554123a62e9ac77107cb20e4f6cc503af4" CACHE STRING "llama.cpp repository commit")
message(STATUS "Using llama.cpp tag ${LLAMACPP_COMMIT}")

# Fetch llama.cpp latest
# FIXME: Maybe use a vendored llama.cpp build for stability
include(FetchContent)
FetchContent_Declare(
    llama
    GIT_REPOSITORY ${LLAMACPP_REPO}
    GIT_TAG ${LLAMACPP_COMMIT}
)

# Required for proper builds
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Disable unused components
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama.cpp: build examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama.cpp: build tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama.cpp: build server" FORCE)

FetchContent_MakeAvailable(llama)

# Apple build changes
# From llama-cpp-python
if (APPLE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    # Need to disable these llama.cpp flags on Apple x86_64,
    # otherwise users may encounter invalid instruction errors
    set(GGML_AVX "Off" CACHE BOOL "ggml: enable AVX" FORCE)
    set(GGML_AVX2 "Off" CACHE BOOL "ggml: enable AVX2" FORCE)
    set(GGML_FMA "Off" CACHE BOOL "gml: enable FMA" FORCE)
    set(GGML_F16C "Off" CACHE BOOL "gml: enable F16C" FORCE)
endif()

if (APPLE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "llama: embed metal library" FORCE)
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
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../lib"
)

add_executable(minimal_cpp_test
    minimal_cpp_test.cpp
)

target_link_libraries(minimal_cpp_test PRIVATE llama)

# Windows options
if(WIN32)
    set_target_properties(deno_cpp_binding PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )
endif()
