if (Get-Command cmake -ErrorAction SilentlyContinue) {
    Write-Host "Found CMake: $(cmake --version)"
} else {
    Import-Module 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
    Enter-VsDevShell -VsInstallPath 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools' -DevCmdArguments '-arch=x64 -host_arch=x64'
}

if (Test-Path env:CUDA_PATH) {
    $extraCmakeArgs += @(
        "-DGGML_CUDA=ON",
        "-DCMAKE_GENERATOR_TOOLSET='cuda=$env:CUDA_PATH'"
    )
}

cmake . -B build -DCMAKE_BUILD_TYPE=Release $extraCmakeArgs
cmake --build build --config Release --target deno_cpp_binding
cp build/Release/*.dll ../lib
cp build/bin/Release/*.dll ../lib
