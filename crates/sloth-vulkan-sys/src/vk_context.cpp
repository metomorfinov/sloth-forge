#include "vk_context.h"
#include "sloth_vulkan.h"

#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>

namespace sloth {

VulkanContext& VulkanContext::instance() {
    static VulkanContext ctx;
    return ctx;
}

VulkanContext::~VulkanContext() {
    shutdown();
}

int VulkanContext::init(int prefer_discrete, char* out_device_name, size_t max_len) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        if (out_device_name && max_len > 0) {
            strncpy(out_device_name, device_info_.device_name.c_str(), max_len - 1);
            out_device_name[max_len - 1] = '\0';
        }
        return SLOTH_VK_SUCCESS;
    }

    // 1. Create Vulkan Instance
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "SlothForge Compute Engine";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "SlothVulkan";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> instance_extensions;
    uint32_t extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
    if (extension_count > 0) {
        std::vector<VkExtensionProperties> available_exts(extension_count);
        vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available_exts.data());
        for (const auto& ext : available_exts) {
            if (strcmp(ext.extensionName, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0) {
                instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
                break;
            }
        }
    }

    VkInstanceCreateInfo instance_ci{};
    instance_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_ci.pApplicationInfo = &app_info;
    instance_ci.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
    instance_ci.ppEnabledExtensionNames = instance_extensions.empty() ? nullptr : instance_extensions.data();

    VkResult res = vkCreateInstance(&instance_ci, nullptr, &instance_);
    if (res != VK_SUCCESS) {
        // Fallback to API version 1.2 if 1.3 failed
        app_info.apiVersion = VK_API_VERSION_1_2;
        res = vkCreateInstance(&instance_ci, nullptr, &instance_);
        if (res != VK_SUCCESS) {
            std::cerr << "[SlothVulkan] Failed to create Vulkan instance: " << res << std::endl;
            return SLOTH_VK_ERROR_INIT_FAILED;
        }
    }

    // 2. Select Physical Device
    int select_res = select_physical_device(prefer_discrete);
    if (select_res != SLOTH_VK_SUCCESS) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
        return select_res;
    }

    // 3. Create Logical Device & Compute Queue
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci{};
    queue_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = compute_queue_family_;
    queue_ci.queueCount = 1;
    queue_ci.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures device_features{};

    VkDeviceCreateInfo device_ci{};
    device_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_ci.queueCreateInfoCount = 1;
    device_ci.pQueueCreateInfos = &queue_ci;
    device_ci.pEnabledFeatures = &device_features;

    res = vkCreateDevice(physical_device_, &device_ci, nullptr, &device_);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to create logical device: " << res << std::endl;
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
        return SLOTH_VK_ERROR_INIT_FAILED;
    }

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

    // 4. Create Command Pool for Compute / Transfer operations
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_ci.queueFamilyIndex = compute_queue_family_;

    res = vkCreateCommandPool(device_, &pool_ci, nullptr, &command_pool_);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to create command pool: " << res << std::endl;
        vkDestroyDevice(device_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        device_ = VK_NULL_HANDLE;
        instance_ = VK_NULL_HANDLE;
        return SLOTH_VK_ERROR_INIT_FAILED;
    }

    // Query physical device memory properties
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties_);

    // Fill output device name
    if (out_device_name && max_len > 0) {
        strncpy(out_device_name, device_info_.device_name.c_str(), max_len - 1);
        out_device_name[max_len - 1] = '\0';
    }

    initialized_ = true;
    return SLOTH_VK_SUCCESS;
}

int VulkanContext::select_physical_device(int prefer_discrete) {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        std::cerr << "[SlothVulkan] No Vulkan physical devices found." << std::endl;
        return SLOTH_VK_ERROR_NO_DEVICE;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    VkPhysicalDevice best_device = VK_NULL_HANDLE;
    int64_t best_score = -1;
    uint32_t best_compute_queue = 0;
    DeviceInfo best_info{};

    for (const auto& dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);

        // Find compute queue family
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_family_count, queue_families.data());

        int compute_index = -1;
        // First look for a dedicated compute queue (COMPUTE without GRAPHICS)
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                !(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                compute_index = static_cast<int>(i);
                break;
            }
        }
        // Fallback to any compute queue
        if (compute_index == -1) {
            for (uint32_t i = 0; i < queue_family_count; ++i) {
                if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    compute_index = static_cast<int>(i);
                    break;
                }
            }
        }

        if (compute_index == -1) {
            continue; // Device cannot run compute
        }

        // Calculate VRAM size
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(dev, &mem_props);
        uint64_t total_vram = 0;
        for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
            if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                total_vram += mem_props.memoryHeaps[i].size;
            }
        }

        int64_t score = 0;
        std::string name(props.deviceName);

        // High priority for AMD Radeon RX 570 / POLARIS10
        if (props.vendorID == 0x1002) { // AMD
            score += 100000;
        }
        if (name.find("RX 570") != std::string::npos ||
            name.find("POLARIS10") != std::string::npos ||
            name.find("Polaris10") != std::string::npos) {
            score += 1000000;
        }

        if (prefer_discrete && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 50000;
        } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 10000;
        }

        // Add VRAM in MB to score
        score += static_cast<int64_t>(total_vram / (1024 * 1024));

        if (score > best_score) {
            best_score = score;
            best_device = dev;
            best_compute_queue = static_cast<uint32_t>(compute_index);

            best_info.device_name = props.deviceName;
            best_info.vendor_id = props.vendorID;
            best_info.device_id = props.deviceID;
            best_info.device_type = props.deviceType;
            best_info.total_vram_bytes = total_vram;
            best_info.compute_queue_family_index = best_compute_queue;
            best_info.subgroup_size = 64; // Default on AMD Polaris
        }
    }

    if (best_device == VK_NULL_HANDLE) {
        std::cerr << "[SlothVulkan] No suitable compute GPU found." << std::endl;
        return SLOTH_VK_ERROR_NO_DEVICE;
    }

    physical_device_ = best_device;
    compute_queue_family_ = best_compute_queue;
    device_info_ = best_info;

    return SLOTH_VK_SUCCESS;
}

void VulkanContext::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) return;

    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        if (command_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
            command_pool_ = VK_NULL_HANDLE;
        }
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    allocated_vram_bytes_.store(0);
    initialized_ = false;
}

uint32_t VulkanContext::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const {
    for (uint32_t i = 0; i < mem_properties_.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (mem_properties_.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    // Fallback: If exact match not found for device local, find any matching type filter
    for (uint32_t i = 0; i < mem_properties_.memoryTypeCount; ++i) {
        if (type_filter & (1 << i)) {
            return i;
        }
    }
    return UINT32_MAX;
}

void VulkanContext::get_vram_info(uint64_t* total, uint64_t* used, uint64_t* free) const {
    uint64_t total_vram = 0;
    for (uint32_t i = 0; i < mem_properties_.memoryHeapCount; ++i) {
        if (mem_properties_.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total_vram += mem_properties_.memoryHeaps[i].size;
        }
    }

    uint64_t used_vram = allocated_vram_bytes_.load(std::memory_order_relaxed);
    uint64_t free_vram = (total_vram > used_vram) ? (total_vram - used_vram) : 0;

    if (total) *total = total_vram;
    if (used) *used = used_vram;
    if (free) *free = free_vram;
}

VkCommandBuffer VulkanContext::begin_single_time_commands() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool_;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device_, &alloc_info, &cmd);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd, &begin_info);
    return cmd;
}

void VulkanContext::end_single_time_commands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(device_, &fence_ci, nullptr, &fence);

    vkQueueSubmit(compute_queue_, 1, &submit_info, fence);
    vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}

} // namespace sloth
