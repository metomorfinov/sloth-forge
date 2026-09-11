CXX ?= clang++
GLSLANG ?= /usr/bin/glslangValidator
CXXFLAGS ?= -std=c++20 -O3 -fPIC -I./include -Wall -Wextra -DSLOTH_VK_BUILD_SHARED
LDFLAGS ?= -shared -lvulkan

SHADER_DIR = shaders
SRC_DIR = src
TEST_DIR = tests

SHADERS = $(SHADER_DIR)/gemm_f32.spv \
          $(SHADER_DIR)/lora_forward.spv \
          $(SHADER_DIR)/lora_backward.spv \
          $(SHADER_DIR)/rmsnorm.spv \
          $(SHADER_DIR)/adamw.spv

SOURCES = $(SRC_DIR)/vk_context.cpp \
          $(SRC_DIR)/vk_buffer.cpp \
          $(SRC_DIR)/vk_pipeline.cpp \
          $(SRC_DIR)/sloth_vulkan_api.cpp

TARGET = libsloth_vulkan.so
TEST_BIN = test_vulkan

.PHONY: all shaders embed lib test clean

all: shaders embed $(TARGET)

shaders: $(SHADERS)

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%.comp
	$(GLSLANG) -V $< -o $@

embed: $(SHADERS)
	python3 -c "import os; shaders = ['gemm_f32', 'lora_forward', 'lora_backward', 'rmsnorm', 'adamw']; \
	with open('$(SRC_DIR)/spv_embedded.h', 'w') as out: \
	    out.write('#ifndef SLOTH_SPV_EMBEDDED_H\n#define SLOTH_SPV_EMBEDDED_H\n\n#include <cstdint>\n#include <cstddef>\n\nnamespace sloth {\n'); \
	    for s in shaders: \
	        data = open(f'$(SHADER_DIR)/{s}.spv', 'rb').read(); \
	        out.write(f'inline const size_t spv_{s}_size = {len(data)};\n'); \
	        out.write(f'inline const uint32_t spv_{s}_data[] = {{\n'); \
	        words = [int.from_bytes(data[i:i+4], \"little\") for i in range(0, len(data), 4)]; \
	        for i in range(0, len(words), 8): \
	            out.write('    ' + ', '.join(f'0x{w:08x}u' for w in words[i:i+8]) + ',\n'); \
	        out.write('};\n\n'); \
	    out.write('} // namespace sloth\n#endif // SLOTH_SPV_EMBEDDED_H\n')"

$(TARGET): $(SOURCES) $(SRC_DIR)/spv_embedded.h
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $@

test: $(TARGET) $(TEST_DIR)/test_vulkan.cpp
	$(CXX) -std=c++20 -O3 -I./include $(TEST_DIR)/test_vulkan.cpp -L. -lsloth_vulkan -lvulkan -Wl,-rpath,. -o $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(SHADER_DIR)/*.spv $(TARGET) $(TEST_BIN) $(SRC_DIR)/spv_embedded.h
