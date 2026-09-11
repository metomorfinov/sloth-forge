#ifndef SLOTH_VK_PIPELINE_H
#define SLOTH_VK_PIPELINE_H

#include <vulkan/vulkan.h>
#include "vk_buffer.h"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace sloth {

class ComputePipeline {
public:
    ComputePipeline(
        const std::string& shader_name,
        uint32_t num_storage_buffers,
        uint32_t push_constant_size = 0
    );
    ~ComputePipeline();

    bool init();
    void destroy();

    int dispatch(
        uint32_t group_x,
        uint32_t group_y,
        uint32_t group_z,
        const void* push_constants,
        size_t push_size,
        const std::vector<std::shared_ptr<SlothBuffer>>& buffers
    );

    const std::string& name() const { return shader_name_; }

private:
    std::string shader_name_;
    uint32_t num_buffers_ = 0;
    uint32_t push_constant_size_ = 0;

    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet desc_set_ = VK_NULL_HANDLE;

    std::vector<uint32_t> load_spirv(const std::string& name);
};

class PipelineManager {
public:
    static PipelineManager& instance();

    std::shared_ptr<ComputePipeline> get_pipeline(
        const std::string& name,
        uint32_t num_buffers,
        uint32_t push_constant_size
    );

    void destroy_all();

private:
    PipelineManager() = default;
    ~PipelineManager();
    PipelineManager(const PipelineManager&) = delete;
    PipelineManager& operator=(const PipelineManager&) = delete;

    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<ComputePipeline>> pipelines_;
};

} // namespace sloth

#endif // SLOTH_VK_PIPELINE_H
