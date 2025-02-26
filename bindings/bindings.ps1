if (Get-Command cmake -ErrorAction SilentlyContinue) {
    Write-Host "Found CMake: $(cmake --version)"
} else {
    Import-Module 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
    Enter-VsDevShell -VsInstallPath 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools' -DevCmdArguments '-arch=x64 -host_arch=x64'
}

if ($env:GGML_CUDA -eq 1) {
    Write-Host "CUDA enabled, including in build"

    $extraCmakeArgs += @(
        "-DGGML_CUDA=ON"
    )
}

cmake . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release $extraCmakeArgs
cmake --build build --config Release --target deno_cpp_binding -j $env:NUMBER_OF_PROCESSORS
Copy-Item build/*.dll ../lib
Copy-Item build/bin/*.dll ../lib
