#ifndef SLOTH_VK_CONTEXT_H
#define SLOTH_VK_CONTEXT_H

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <atomic>

namespace sloth {

struct DeviceInfo {
    std::string device_name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    VkPhysicalDeviceType device_type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    uint64_t total_vram_bytes = 0;
    uint32_t compute_queue_family_index = 0;
    uint32_t subgroup_size = 64;
};

class VulkanContext {
public:
    static VulkanContext& instance();

    int init(int prefer_discrete, char* out_device_name, size_t max_len);
    void shutdown();

    bool is_initialized() const { return initialized_; }

    VkInstance get_instance() const { return instance_; }
    VkPhysicalDevice get_physical_device() const { return physical_device_; }
    VkDevice get_device() const { return device_; }
    VkQueue get_compute_queue() const { return compute_queue_; }
    uint32_t get_compute_family() const { return compute_queue_family_; }
    VkCommandPool get_command_pool() const { return command_pool_; }
    const DeviceInfo& get_device_info() const { return device_info_; }

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;
    void get_vram_info(uint64_t* total, uint64_t* used, uint64_t* free) const;

    void add_allocated_vram(uint64_t bytes) {
        allocated_vram_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    void sub_allocated_vram(uint64_t bytes) {
        allocated_vram_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    }

    VkCommandBuffer begin_single_time_commands();
    void end_single_time_commands(VkCommandBuffer cmd);

private:
    VulkanContext() = default;
    ~VulkanContext();
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    int select_physical_device(int prefer_discrete);

    bool initialized_ = false;
    mutable std::mutex mutex_;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_ = 0;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;

    DeviceInfo device_info_;
    VkPhysicalDeviceMemoryProperties mem_properties_{};
    std::atomic<uint64_t> allocated_vram_bytes_{0};
};

} // namespace sloth

#endif // SLOTH_VK_CONTEXT_H
