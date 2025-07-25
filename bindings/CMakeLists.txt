cmake_minimum_required(VERSION 3.14.0)
project(LlamaMultiUserInference)
set(CMAKE_CXX_STANDARD 17)

option(LLGUIDANCE "Enable LLGuidance support (requires Rust)" OFF)

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

set(USE_CUSTOM_LLAMA OFF)

# Do not cache these variables with subsequent builds
if (DEFINED LLAMACPP_REPO)
    set(USE_CUSTOM_LLAMA ON)
else()
    set(LLAMACPP_REPO "https://github.com/ggerganov/llama.cpp.git")
endif()

# Stable llama.cpp commit for bindings
if (DEFINED LLAMACPP_COMMIT)
    set(USE_CUSTOM_LLAMA ON)
else()
    set(LLAMACPP_COMMIT "66906cd82a4a1fd10151707cee3f66cb61fc4055")
endif()

if (USE_CUSTOM_LLAMA)
    message(STATUS "Using a custom commit or repo for llama.cpp. Build might not work as expected. Here be dragons!")
endif()

message(STATUS "Using llama.cpp repo ${LLAMACPP_REPO}")
message(STATUS "Using llama.cpp tag ${LLAMACPP_COMMIT}")

# Optional: You can also enable mixed FP16/FP32 computation for faster processing
# set(LLAMA_CUDA_F16 ON CACHE BOOL "llama.cpp: use float16 for GPU operations" FORCE)
# set(GGML_CUDA ON CACHE BOOL "llama.cpp: use float16 for GPU operations" FORCE)

# Disable unused components to speed up build
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama.cpp: build examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama.cpp: build tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama.cpp: build server" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "llama.cpp: use libcurl" FORCE)

# Enable common
set(LLAMA_BUILD_COMMON ON CACHE BOOL "llama.cpp: build common utils library" FORCE)

if(LLGUIDANCE)
    find_program(CARGO cargo)
    if(CARGO)
        message(STATUS "Including LLGuidance in build")
        set(LLAMA_LLGUIDANCE ON CACHE BOOL "llama.cpp: enable LLGuidance support" FORCE)
    else()
        message(FATAL_ERROR "LLGuidance is enabled, but requires Rust for compilation. Get it at https://rustup.rs")
    endif()
else()
    message(STATUS "LLGuidance support is disabled. Enable with -DLLGUIDANCE=ON for grammar, JSON schema, and regex support.")
    set(LLAMA_LLGUIDANCE OFF CACHE BOOL "llama.cpp: disable LLGuidance support" FORCE)
endif()

# Fetch llama.cpp latest
# FIXME: Maybe use a vendored llama.cpp build for stability
include(FetchContent)
FetchContent_Declare(
    llama
    GIT_REPOSITORY ${LLAMACPP_REPO}
    GIT_TAG ${LLAMACPP_COMMIT}
)

# Set build type to Release for performance
set(CMAKE_BUILD_TYPE Release)

# Build all libs to bin
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Make llama.cpp available
FetchContent_MakeAvailable(llama)

message(STATUS "llama source dir: ${llama_SOURCE_DIR}")

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

# Create a library from c_library.cpp
add_library(c_library SHARED
    server/c_library.cpp
)

# Set include directories for the library
target_include_directories(c_library PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/server
    ${llama_SOURCE_DIR}/src
)

# Link llama libraries to our c_library
target_link_libraries(c_library PUBLIC llama common)

# Create our main executable
add_executable(multi_user_inference
    server/server_basic_example.cpp
)

# set_target_properties(multi_user_inference PROPERTIES
#     INSTALL_RPATH "${CMAKE_BINARY_DIR}/bin"
# )

# Include directories for main executable
target_include_directories(multi_user_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/server
)

# Link our c_library to the main executable
target_link_libraries(multi_user_inference PRIVATE
    c_library
)

if(LLGUIDANCE)
    target_compile_definitions(c_library PUBLIC LLGUIDANCE_BUILT=1)
endif()

# Windows options
if(WIN32)
    set_target_properties(c_library PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )
endif()
