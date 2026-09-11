param (
    [switch]$CompileShadersOnly
)

$ErrorActionPreference = "Stop"
Write-Host "[*] Compiling Vulkan GLSL shaders to SPIR-V..." -ForegroundColor Cyan

$ShaderDir = "$PSScriptRoot\shaders"
$Shaders = Get-ChildItem -Path $ShaderDir -Filter "*.comp"

foreach ($Shader in $Shaders) {
    $OutputFile = "$ShaderDir\$($Shader.BaseName).spv"
    Write-Host "  -> Compiling $($Shader.Name) to $($Shader.BaseName).spv"
    glslangValidator -V "$($Shader.FullName)" -o "$OutputFile"
}

Write-Host "[+] All SPIR-V compute shaders compiled successfully." -ForegroundColor Green

if ($CompileShadersOnly) {
    exit 0
}

Write-Host "[*] Compiling C++20 Vulkan shared library for Windows..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path "$PSScriptRoot\bin" -Force | Out-Null

cl.exe /O2 /EHsc /std:c++20 /LD `
    "$PSScriptRoot\src\vk_context.cpp" `
    "$PSScriptRoot\src\vk_buffer.cpp" `
    "$PSScriptRoot\src\vk_pipeline.cpp" `
    "$PSScriptRoot\src\sloth_vulkan_api.cpp" `
    /I"$PSScriptRoot\include" `
    /I"$env:VULKAN_SDK\Include" `
    /link /LIBPATH:"$env:VULKAN_SDK\Lib" vulkan-1.lib `
    /OUT:"$PSScriptRoot\bin\libsloth_vulkan.dll"

Write-Host "[+] libsloth_vulkan.dll compiled successfully." -ForegroundColor Green
