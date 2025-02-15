if (Get-Command cmake -ErrorAction SilentlyContinue) {
    Write-Host "Found CMake: $(cmake --version)"
} else {
    Import-Module 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
    Enter-VsDevShell -VsInstallPath 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools' -DevCmdArguments '-arch=x64 -host_arch=x64'
}

if (Test-Path env:CUDA_PATH) {
    $extraCmakeArgs += @(
        "-DGGML_CUDA=ON"
    )
}

cmake . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release $extraCmakeArgs
cmake --build build --config Release --target deno_cpp_binding\
Copy-File build/*.dll ../lib
Copy-File build/bin/*.dll ../lib
