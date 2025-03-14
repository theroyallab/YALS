# Build Instructions

YALS contains two components:
1. TypeScript code: Universally buildable on any OS
2. C++ bindings: Requires an OS-specific C++ compiler and additional setup

The C++ bindings need to be built to integrate the `llama.cpp` library and provide the necessary "glue" required by YALS.

## Prerequisites

To get started, install the following prerequisites:
- [Deno](https://deno.com) 
- A C++ compiler:
	- Windows: Visual Studio 2022 build tools
	- macOS: Xcode command-line tools (`xcode-select --install`)
	- Linux: GCC (`sudo apt install build-essential`)
- CMake:
	- Windows: Installed with Visual Studio build tools
	- macOS (homebrew): `brew install cmake`
	- Linux: `sudo apt install cmake` (For Ubuntu 22.04, follow this [askubuntu](https://askubuntu.com/a/865294) answer to install the latest version)
- Ninja (Makes builds faster)
	- Windows: `winget install -e --id Ninja-build.Ninja`
	- macOS (homebrew): `brew install ninja`
	- Linux: `sudo apt install ninja-build`
- [sccache](https://github.com/mozilla/sccache) (optional, but speeds up subsequent builds)

## Building

Clone the repository and navigate to the project folder:
```sh
git clone https://github.com/theroyallab/YALS.git
cd YALS
```

All build commands are encapsulated in Deno tasks, similar to npm scripts in NodeJS.

To build the C++bindings:

- Windows: `deno task bindings-win`
- macOS/Linux: `deno task bindings`

This will invoke CMake to build the bindings and copy the resulting shared libraries to the `lib` folder.

Optionally, environment variables can be set for certain architectures when building (ex. CUDA):
- `MAX_JOBS`: Number of parallel jobs (defaults to the number of CPU cores)
- `LLAMACPP_REPO`: Point to a custom repository for llama.cpp (Here be dragons!)
- `LLAMACPP_TAG`: Set a specific tag for llama.cpp (Here be dragons!)
- `GGML_CUDA=1`: Enables CUDA support
- `CMAKE_CUDA_ARCHITECTURES`: Specifies CUDA compute capabilities (defaults to `native` if using CMake > 3.24)
- `GGML_VULKAN=1`: Enables Vulkan Support
- `GGML_HIP=1`: Enables HIP ROCM Support (Requires specifying DAMDGPU_TARGETS, Linux only)
- `AMDGPU_TARGETS`: Specify ROCM target (example: `gfx1030`)

## Running

To start the server with necessary permissions:
```sh
deno task start
```

With full permissions (useful for testing new features):
```sh
deno run -A main.ts
```

## Packaging

> [!NOTE]
> **Note:** All YALS commits are built via GitHub Actions, so manual packaging is typically unnecessary unless you need to distribute builds with a custom build configuration.

To create a distributable binary:

1. Run: `deno task build` to package all TypeScript code into a standalone binary
2. Zip the following files and directories:
   - `YALS(.exe)`
   - `lib/`
   - `models/`
   - `templates/`
   - `config_sample.yml`
3. Distribute the archive, and the recipient can simply extract and run it.
