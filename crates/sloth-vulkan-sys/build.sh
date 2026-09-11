#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=== [1/4] Compiling GLSL Compute Shaders with glslangValidator ==="
for shader in gemm_f32 lora_forward lora_backward rmsnorm adamw; do
    echo "  Compiling shaders/${shader}.comp -> shaders/${shader}.spv..."
    /usr/bin/glslangValidator -V "shaders/${shader}.comp" -o "shaders/${shader}.spv"
done

echo "=== [2/4] Generating Embedded SPIR-V Header ==="
python3 -c "
shaders = ['gemm_f32', 'lora_forward', 'lora_backward', 'rmsnorm', 'adamw']
with open('src/spv_embedded.h', 'w') as out:
    out.write('#ifndef SLOTH_SPV_EMBEDDED_H\n#define SLOTH_SPV_EMBEDDED_H\n\n#include <cstdint>\n#include <cstddef>\n\nnamespace sloth {\n')
    for s in shaders:
        with open(f'shaders/{s}.spv', 'rb') as f:
            data = f.read()
        out.write(f'inline const size_t spv_{s}_size = {len(data)};\n')
        out.write(f'inline const uint32_t spv_{s}_data[] = {{\n')
        words = [int.from_bytes(data[i:i+4], \"little\") for i in range(0, len(data), 4)]
        for i in range(0, len(words), 8):
            out.write('    ' + ', '.join(f'0x{w:08x}u' for w in words[i:i+8]) + ',\n')
        out.write('};\n\n')
    out.write('} // namespace sloth\n#endif // SLOTH_SPV_EMBEDDED_H\n')
"

echo "=== [3/4] Building libsloth_vulkan.so with clang++ ==="
clang++ -shared -fPIC -O3 -std=c++20 -I./include -DSLOTH_VK_BUILD_SHARED \
    src/vk_context.cpp \
    src/vk_buffer.cpp \
    src/vk_pipeline.cpp \
    src/sloth_vulkan_api.cpp \
    -lvulkan -o libsloth_vulkan.so

echo "=== [4/4] Successfully built libsloth_vulkan.so ==="
ls -lh libsloth_vulkan.so
