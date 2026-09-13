#include "vk_pipeline.h"
#include "vk_context.h"
#include "sloth_vulkan.h"
// build.rs компилирует шейдеры из shaders/*.comp при каждой сборке, если найден glslangValidator;
// иначе используются копии, сохранённые в репозитории (src/spv_embedded.h).
#ifdef SLOTH_VK_GENERATED_SPV
#include <spv_embedded_generated.h>
#else
#include "spv_embedded.h"
#endif

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>

namespace sloth {

namespace {

// Для разработки шейдеров: папка с .spv, которые подменяют встроенные копии
const char* const SHADER_DIR_ENV = "SLOTH_SHADER_DIR";
const uint32_t SPIRV_MAGIC = 0x07230203;
const size_t SPIRV_WORD_BYTES = 4;

struct EmbeddedShader {
    const char* name;
    const uint32_t* words;
    size_t size_bytes;
};

const std::array<EmbeddedShader, 5> EMBEDDED_SHADERS = {{
    {"gemm_f32", spv_gemm_f32_data, spv_gemm_f32_size},
    {"lora_forward", spv_lora_forward_data, spv_lora_forward_size},
    {"lora_backward", spv_lora_backward_data, spv_lora_backward_size},
    {"rmsnorm", spv_rmsnorm_data, spv_rmsnorm_size},
    {"adamw", spv_adamw_data, spv_adamw_size},
}};

// Пустой результат — файла нет или это не SPIR-V.
std::vector<uint32_t> read_spirv_file(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) return {};
    const auto file_size = static_cast<size_t>(file.tellg());
    if (file_size == 0 || file_size % SPIRV_WORD_BYTES != 0) return {};
    std::vector<uint32_t> words(file_size / SPIRV_WORD_BYTES);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(file_size));
    if (!file || words[0] != SPIRV_MAGIC) return {};
    return words;
}

} // namespace

ComputePipeline::ComputePipeline(
    const std::string& shader_name,
    uint32_t num_storage_buffers,
    uint32_t push_constant_size
) : shader_name_(shader_name),
    num_buffers_(num_storage_buffers),
    push_constant_size_(push_constant_size) {
}

ComputePipeline::~ComputePipeline() {
    destroy();
}

std::vector<uint32_t> ComputePipeline::load_spirv(const std::string& name) {
    // Раньше .spv искались в папках относительно текущего каталога (shaders/, ../shaders/) и
    // перекрывали встроенные копии: устаревший файл рядом с программой молча менял расчёты.
    // Теперь с диска шейдеры берутся только по явной просьбе через SLOTH_SHADER_DIR.
    if (const char* dir = std::getenv(SHADER_DIR_ENV)) {
        const std::string path = std::string(dir) + "/" + name + ".spv";
        std::vector<uint32_t> words = read_spirv_file(path);
        if (words.empty()) {
            std::cerr << "[SlothVulkan] " << SHADER_DIR_ENV << " is set, but " << path
                      << " is missing or is not SPIR-V." << std::endl;
        } else {
            std::cerr << "[SlothVulkan] Using shader from " << path << std::endl;
        }
        return words;
    }

    for (const auto& shader : EMBEDDED_SHADERS) {
        if (name == shader.name) {
            return std::vector<uint32_t>(shader.words, shader.words + shader.size_bytes / SPIRV_WORD_BYTES);
        }
    }
    std::cerr << "[SlothVulkan] Unknown shader: " << name << std::endl;
    return {};
}

bool ComputePipeline::init() {
    auto& ctx = VulkanContext::instance();
    if (!ctx.is_initialized()) return false;
    VkDevice device = ctx.get_device();

    std::vector<uint32_t> spirv = load_spirv(shader_name_);
    if (spirv.empty()) return false;

    // 1. Shader Module
    VkShaderModuleCreateInfo module_ci{};
    module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_ci.codeSize = spirv.size() * sizeof(uint32_t);
    module_ci.pCode = spirv.data();

    VkResult res = vkCreateShaderModule(device, &module_ci, nullptr, &shader_module_);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to create shader module for " << shader_name_ << ": " << res << std::endl;
        return false;
    }

    // 2. Descriptor Set Layout
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_buffers_);
    for (uint32_t i = 0; i < num_buffers_; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo desc_layout_ci{};
    desc_layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    desc_layout_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    desc_layout_ci.pBindings = bindings.empty() ? nullptr : bindings.data();

    res = vkCreateDescriptorSetLayout(device, &desc_layout_ci, nullptr, &desc_layout_);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to create descriptor set layout: " << res << std::endl;
        return false;
    }

    // 3. Pipeline Layout
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_constant_size_;

    VkPipelineLayoutCreateInfo pipe_layout_ci{};
    pipe_layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipe_layout_ci.setLayoutCount = 1;
    pipe_layout_ci.pSetLayouts = &desc_layout_;
    if (push_constant_size_ > 0) {
        pipe_layout_ci.pushConstantRangeCount = 1;
        pipe_layout_ci.pPushConstantRanges = &push_range;
    }

    res = vkCreatePipelineLayout(device, &pipe_layout_ci, nullptr, &pipeline_layout_);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to create pipeline layout: " << res << std::endl;
        return false;
    }

    // 4. Compute Pipeline
    VkComputePipelineCreateInfo compute_pipe_ci{};
    compute_pipe_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_pipe_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute_pipe_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute_pipe_ci.stage.module = shader_module_;
    compute_pipe_ci.stage.pName = "main";
    compute_pipe_ci.layout = pipeline_layout_;

    res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_pipe_ci, nullptr, &pipeline_);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to create compute pipeline: " << res << std::endl;
        return false;
    }

    // 5. Descriptor Pool and Descriptor Set
    if (num_buffers_ > 0) {
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = num_buffers_;

        VkDescriptorPoolCreateInfo pool_ci{};
        pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_ci.maxSets = 1;
        pool_ci.poolSizeCount = 1;
        pool_ci.pPoolSizes = &pool_size;

        res = vkCreateDescriptorPool(device, &pool_ci, nullptr, &desc_pool_);
        if (res != VK_SUCCESS) {
            std::cerr << "[SlothVulkan] Failed to create descriptor pool: " << res << std::endl;
            return false;
        }

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = desc_pool_;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &desc_layout_;

        res = vkAllocateDescriptorSets(device, &alloc_info, &desc_set_);
        if (res != VK_SUCCESS) {
            std::cerr << "[SlothVulkan] Failed to allocate descriptor set: " << res << std::endl;
            return false;
        }
    }

    return true;
}

void ComputePipeline::destroy() {
    auto& ctx = VulkanContext::instance();
    if (!ctx.is_initialized()) return;
    VkDevice device = ctx.get_device();

    if (desc_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, desc_pool_, nullptr);
        desc_pool_ = VK_NULL_HANDLE;
        desc_set_ = VK_NULL_HANDLE;
    }
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (desc_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, desc_layout_, nullptr);
        desc_layout_ = VK_NULL_HANDLE;
    }
    if (shader_module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, shader_module_, nullptr);
        shader_module_ = VK_NULL_HANDLE;
    }
}

int ComputePipeline::dispatch(
    uint32_t group_x,
    uint32_t group_y,
    uint32_t group_z,
    const void* push_constants,
    size_t push_size,
    const std::vector<std::shared_ptr<SlothBuffer>>& buffers
) {
    if (group_x == 0 || group_y == 0 || group_z == 0) {
        return SLOTH_VK_SUCCESS;
    }

    auto& ctx = VulkanContext::instance();
    if (!ctx.is_initialized()) return SLOTH_VK_ERROR_NOT_INITIALIZED;

    if (buffers.size() != num_buffers_) {
        std::cerr << "[SlothVulkan] Buffer count mismatch for pipeline " << shader_name_
                  << ": expected " << num_buffers_ << ", got " << buffers.size() << std::endl;
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }
    if (push_size > push_constant_size_) {
        std::cerr << "[SlothVulkan] Push constants too large for pipeline " << shader_name_ << std::endl;
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }

    // Больше групп, чем разрешает драйвер (на RX 570 — 65535 по каждой оси), — ошибка
    // вызывающего, а не молчаливо пропущенная работа
    const auto& max_groups = ctx.get_device_info().max_compute_work_group_count;
    if (group_x > max_groups[0] || group_y > max_groups[1] || group_z > max_groups[2]) {
        std::cerr << "[SlothVulkan] Dispatch " << group_x << "x" << group_y << "x" << group_z
                  << " exceeds device limit " << max_groups[0] << "x" << max_groups[1] << "x"
                  << max_groups[2] << " for pipeline " << shader_name_ << std::endl;
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }

    std::vector<VkDescriptorBufferInfo> buf_infos(num_buffers_);
    for (uint32_t i = 0; i < num_buffers_; ++i) {
        if (!buffers[i] || buffers[i]->buffer == VK_NULL_HANDLE) {
            return SLOTH_VK_ERROR_INVALID_PARAM;
        }
        buf_infos[i].buffer = buffers[i]->buffer;
        buf_infos[i].offset = 0;
        buf_infos[i].range = (buffers[i]->size > 0) ? buffers[i]->size : VK_WHOLE_SIZE;
    }

    // Набор дескрипторов один на конвейер: обновляем его под общей блокировкой отправки,
    // иначе два потока переписали бы привязки буферов друг у друга
    return ctx.run_commands([&](VkCommandBuffer cmd) {
        if (num_buffers_ > 0) {
            std::vector<VkWriteDescriptorSet> writes(num_buffers_);
            for (uint32_t i = 0; i < num_buffers_; ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = desc_set_;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &buf_infos[i];
            }
            vkUpdateDescriptorSets(ctx.get_device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        if (num_buffers_ > 0) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &desc_set_, 0, nullptr);
        }
        if (push_constants && push_size > 0) {
            vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(push_size), push_constants);
        }
        vkCmdDispatch(cmd, group_x, group_y, group_z);
    });
}

PipelineManager& PipelineManager::instance() {
    static PipelineManager mgr;
    return mgr;
}

PipelineManager::~PipelineManager() {
    destroy_all();
}

std::shared_ptr<ComputePipeline> PipelineManager::get_pipeline(
    const std::string& name,
    uint32_t num_buffers,
    uint32_t push_constant_size
) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) {
        return it->second;
    }

    auto pipe = std::make_shared<ComputePipeline>(name, num_buffers, push_constant_size);
    if (!pipe->init()) {
        std::cerr << "[SlothVulkan] Failed to initialize pipeline: " << name << std::endl;
        return nullptr;
    }

    pipelines_[name] = pipe;
    return pipe;
}

void PipelineManager::destroy_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : pipelines_) {
        pair.second->destroy();
    }
    pipelines_.clear();
}

} // namespace sloth
