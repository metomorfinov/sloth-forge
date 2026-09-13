#ifndef SLOTH_VK_CONTEXT_H
#define SLOTH_VK_CONTEXT_H

#include <vulkan/vulkan.h>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sloth {

struct DeviceInfo {
    std::string device_name;
    // Имя и версия драйвера (VkPhysicalDeviceDriverProperties), например «radv» и «Mesa 26.2.2»
    std::string driver_name;
    std::string driver_info;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t api_version = 0;
    uint32_t driver_version = 0;
    VkPhysicalDeviceType device_type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    uint64_t total_vram_bytes = 0;
    uint32_t compute_queue_family_index = 0;
    // Размер подгруппы от драйвера; 0 — драйвер не сообщает (Vulkan 1.0)
    uint32_t subgroup_size = 0;
    // Наибольшее число рабочих групп в одном vkCmdDispatch по осям X, Y, Z
    uint32_t max_compute_work_group_count[3] = {0, 0, 0};
    // Драйвер умеет VK_EXT_memory_budget: сколько видеопамяти доступно процессу
    bool memory_budget_supported = false;
};

class VulkanContext {
public:
    static VulkanContext& instance();

    int init(int prefer_discrete, char* out_device_name, size_t max_len);
    void shutdown();

    bool is_initialized() const { return initialized_.load(std::memory_order_acquire); }

    VkInstance get_instance() const { return instance_; }
    VkPhysicalDevice get_physical_device() const { return physical_device_; }
    VkDevice get_device() const { return device_; }
    VkQueue get_compute_queue() const { return compute_queue_; }
    uint32_t get_compute_family() const { return compute_queue_family_; }
    VkCommandPool get_command_pool() const { return command_pool_; }
    const DeviceInfo& get_device_info() const { return device_info_; }

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;
    void get_vram_info(uint64_t* total, uint64_t* used, uint64_t* free) const;

    /**
     * Бюджет видеопамяти (VK_EXT_memory_budget) по всем DEVICE_LOCAL-кучам:
     * budget — сколько процесс может занять сейчас с учётом других программ,
     * usage — сколько занимает сам процесс.
     * SLOTH_VK_ERROR_NOT_SUPPORTED, если драйвер расширение не поддерживает.
     */
    int get_memory_budget(uint64_t* budget, uint64_t* usage) const;

    void add_allocated_vram(uint64_t bytes) {
        allocated_vram_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    void sub_allocated_vram(uint64_t bytes) {
        allocated_vram_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    }

    /**
     * Записывает команды через record, отправляет их в очередь и ждёт выполнения.
     *
     * Пул команд, очередь и наборы дескрипторов Vulkan требуют внешней синхронизации,
     * поэтому все отправки идут по одной под общей блокировкой: record можно использовать
     * и для vkUpdateDescriptorSets. Каждый вызов Vulkan проверяется, ошибка возвращается
     * кодом SLOTH_VK_ERROR_*, а не превращается в «успех».
     */
    int run_commands(const std::function<void(VkCommandBuffer)>& record);

private:
    VulkanContext() = default;
    ~VulkanContext();
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    int select_physical_device(int prefer_discrete);
    DeviceInfo describe_device(VkPhysicalDevice device, uint32_t compute_family) const;

    std::atomic<bool> initialized_{false};
    // Защищает init/shutdown
    mutable std::mutex mutex_;
    // Защищает пул команд, очередь и наборы дескрипторов (см. run_commands)
    std::mutex command_mutex_;

    VkInstance instance_ = VK_NULL_HANDLE;
    // Версия API экземпляра: от неё зависит, доступен ли vkGetPhysicalDeviceProperties2
    uint32_t instance_api_version_ = VK_API_VERSION_1_0;
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
