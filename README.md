<div align="center">
    <img src="https://github.com/theroyallab/YALS/blob/main/assets/icon.png?raw=true" />
</div>

# YALS

<p align="left">
    <img src="https://img.shields.io/badge/Deno-2.x-blue" alt="Python 3.10, 3.11, and 3.12">
    <a href="/LICENSE">
        <img src="https://img.shields.io/badge/License-AGPLv3-blue.svg" alt="License: AGPL v3"/>
    </a>
    <a href="https://discord.gg/sYQxnuD7Fj">
        <img src="https://img.shields.io/discord/545740643247456267.svg?logo=discord&color=blue" alt="Discord Server"/>
    </a>
</p>

<p align="left">
    <a href="https://ko-fi.com/I2I3BDTSW">
        <img src="https://img.shields.io/badge/Support_on_Ko--fi-FF5E5B?logo=ko-fi&style=for-the-badge&logoColor=white" alt="Support on Ko-Fi">
    </a>
</p>

> [!NOTE]
> 
>  Need help? Join the [Discord Server](https://discord.gg/sYQxnuD7Fj) and get the `Tabby` role. Please be nice when asking questions.

Welcome to YALS, also known as **Y**et **A**nother **L**lamacpp **S**erver.

YALS is a friendly OAI compatible API server built with Deno, Hono, and Zod, designed to facilitate LLM text generation via the [llama.cpp backend](https://github.com/ggml-org/llama.cpp)

## Disclaimer

This project is in an alpha state. There may be bugs, possibly even ones that could cause thermonuclear war. Please note that commits happen frequently, and builds are distributed via CI.

YALS is a hobby project made for a small amount of users. It is not meant to run on production servers. For that, please look at other solutions that support those workloads.

## Why?

The AI space is full of backend projects that wrap llama.cpp, but I felt that something was missing. This led me to create my own backend, one which is extensible, speedy, and as elegant as TabbyAPI, but specifically for llama.cpp and GGUF.

## What about TabbyAPI?

Here are the reasons why I decided to create a separate project instead of integrating llamacpp support into TabbyAPI:

1. **Separation of concerns**: I want TabbyAPI to stay focused on ExLlama, not become a monolithic backend.
2. **Distribution patterns**: Unlike TabbyAPI, llama.cpp backends are often distributed as binaries. Deno’s compile command is vastly superior to PyInstaller, making binary distribution easier.
3. **Dependency hell**: Python’s dependency system is a mess. Adding another layer of abstractions would confuse users further.
4. **New technologies**: Since C++ (via C bindings) is universally compatible via an FFI interface, I wanted to try something new instead of struggling with Python. The main reason for using Deno is because it augments an easy to learn language (TypeScript) with inbuilt tooling and a robust FFI system.
## Getting Started

To get started, download the latest zip from [releases](https://github.com/theroyallab/YALS/releases/latest) that corresponds to your setup.

The currently supported builds via CI are:

- **macOS**: Metal 
- **Windows/Linux**: CPU 
- **Windows/Linux**: CUDA (built for Turing architectures and newer)

> [!NOTE]
> 
>  If your specific setup is not available via CI, you can build locally via the [building guide](https://github.com/theroyallab/YALS/blob/main/BUILDING.md), or request a certain architecture in issues.

Then follow these steps:

1. Extract the zip file
2. Copy `config_sample.yml` to a file called `config.yml`
3. Edit `config.yml` to configure model loading, networking, and other parameters.
	1. All options are commented: **if you're unsure about an option, it's best to leave it unchanged**.
	2. You can also use CLI arguments, similar to TabbyAPI (ex. `--flash-attention true`).
4. Download a `.gguf` model into the `models` directory (or whatever you set your directory to)
	1. If the model is split into multiple parts (`00001-of-0000x.gguf`), set `model_name` in `config.yml` to the **first part** (ending in `00001`). Other parts will load automatically.
5. Start YALS:
	1. Windows: Double click `YALS.exe` or run `.\YALS.exe` from the terminal (recommended)
	2. macOS/Linux: Open a terminal and run `./YALS`
## Features

- OpenAI compatible API
- Loading/unloading models
- Flexible Jinja2 template engine for chat completions that conforms to HuggingFace
- String banning
- Concurrent inference with Hono + async TypeScript
- Robust validation with Zod

More features will be added as the project matures. If something is missing here, PR it in!

## Supported Model Types

Since YALS uses llama.cpp for inference, the only supported model format is GGUF.

If you want to use other model formats such as Exl2, try [tabbyAPI](https://github.com/theroyallab/TabbyAPI)

## Contributing

Use the template when creating issues or pull requests, otherwise the developers may not look at your post.

If you have issues with the project:

- Describe the issue in detail
- If you have a feature request, please indicate it as such.

If you have a Pull Request:

- Describe the pull request in detail, what, and why you are changing something

## Developers and Permissions

Creators/Developers:

- [kingbri](https://github.com/bdashore3) - TypeScript, Deno, and some C++
- [CoffeeVampire](https://github.com/https://github.com/CoffeeVampir3) - Main C++ developer

## Acknowledgements

YALS would not exist without the work of other contributors and FOSS projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Deno](https://deno.com)
- [Hono](https://hono.dev)
- [Zod](https://zod.dev)
- [KoboldCpp](https://github.com/lostruins/koboldcpp)
- [SillyTavern](https://github.com/SillyTavern/SillyTavern)
