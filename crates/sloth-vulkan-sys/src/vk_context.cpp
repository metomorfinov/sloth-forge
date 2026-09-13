#include "vk_context.h"
#include "sloth_vulkan.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace sloth {

namespace {

// Переменная окружения для выбора видеокарты: номер устройства в списке Vulkan или часть имени
const char* const DEVICE_OVERRIDE_ENV = "SLOTH_VK_DEVICE";
const uint32_t PREFERRED_API_VERSION = VK_API_VERSION_1_3;

// Ошибка Vulkan → код SLOTH_VK_ERROR_* с сообщением в stderr.
int report_failure(const char* call, VkResult result) {
    std::cerr << "[SlothVulkan] " << call << " failed: VkResult " << result << std::endl;
    switch (result) {
        case VK_ERROR_OUT_OF_HOST_MEMORY:
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return SLOTH_VK_ERROR_OUT_OF_MEMORY;
        default:
            return SLOTH_VK_ERROR_DISPATCH_FAILED;
    }
}

bool has_device_extension(VkPhysicalDevice device, const char* name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data()) != VK_SUCCESS) {
        return false;
    }
    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& ext) {
        return std::strcmp(ext.extensionName, name) == 0;
    });
}

// Очередь для вычислений: сначала отдельная (без графики), иначе любая с COMPUTE. -1 — нет.
int find_compute_queue_family(VkPhysicalDevice device) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            return static_cast<int>(i);
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

uint64_t device_local_bytes(const VkPhysicalDeviceMemoryProperties& memory) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total += memory.memoryHeaps[i].size;
        }
    }
    return total;
}

// Приоритет типа устройства: чем больше, тем лучше.
int device_type_rank(VkPhysicalDeviceType type, bool prefer_discrete) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return prefer_discrete ? 4 : 3;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return prefer_discrete ? 3 : 4;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
        default: return 0;
    }
}

std::string to_lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

void copy_name(const std::string& name, char* out, size_t max_len) {
    if (!out || max_len == 0) return;
    std::strncpy(out, name.c_str(), max_len - 1);
    out[max_len - 1] = '\0';
}

// Освобождает буфер команд и забор при любом выходе из run_commands.
struct SubmitResources {
    VkDevice device;
    VkCommandPool pool;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    ~SubmitResources() {
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
        if (cmd != VK_NULL_HANDLE) vkFreeCommandBuffers(device, pool, 1, &cmd);
    }
};

} // namespace

VulkanContext& VulkanContext::instance() {
    static VulkanContext ctx;
    return ctx;
}

VulkanContext::~VulkanContext() {
    shutdown();
}

int VulkanContext::init(int prefer_discrete, char* out_device_name, size_t max_len) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_initialized()) {
        copy_name(device_info_.device_name, out_device_name, max_len);
        return SLOTH_VK_SUCCESS;
    }

    // 1. Экземпляр Vulkan. Версию API берём не выше той, что поддерживает загрузчик.
    uint32_t loader_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion(&loader_version) != VK_SUCCESS) {
        loader_version = VK_API_VERSION_1_0;
    }
    instance_api_version_ = std::min(loader_version, PREFERRED_API_VERSION);

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "SlothForge Compute Engine";
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app_info.pEngineName = "SlothVulkan";
    app_info.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app_info.apiVersion = instance_api_version_;

    VkInstanceCreateInfo instance_ci{};
    instance_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_ci.pApplicationInfo = &app_info;

    VkResult res = vkCreateInstance(&instance_ci, nullptr, &instance_);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to create Vulkan instance: " << res << std::endl;
        instance_ = VK_NULL_HANDLE;
        return SLOTH_VK_ERROR_INIT_FAILED;
    }

    // 2. Видеокарта
    int select_res = select_physical_device(prefer_discrete);
    if (select_res != SLOTH_VK_SUCCESS) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
        return select_res;
    }

    // 3. Логическое устройство и очередь вычислений
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci{};
    queue_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = compute_queue_family_;
    queue_ci.queueCount = 1;
    queue_ci.pQueuePriorities = &queue_priority;

    std::vector<const char*> device_extensions;
    if (device_info_.memory_budget_supported) {
        device_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }

    VkPhysicalDeviceFeatures device_features{};
    VkDeviceCreateInfo device_ci{};
    device_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_ci.queueCreateInfoCount = 1;
    device_ci.pQueueCreateInfos = &queue_ci;
    device_ci.pEnabledFeatures = &device_features;
    device_ci.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    device_ci.ppEnabledExtensionNames = device_extensions.empty() ? nullptr : device_extensions.data();

    res = vkCreateDevice(physical_device_, &device_ci, nullptr, &device_);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to create logical device: " << res << std::endl;
        device_ = VK_NULL_HANDLE;
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
        return SLOTH_VK_ERROR_INIT_FAILED;
    }

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

    // 4. Пул команд
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_ci.queueFamilyIndex = compute_queue_family_;

    res = vkCreateCommandPool(device_, &pool_ci, nullptr, &command_pool_);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to create command pool: " << res << std::endl;
        command_pool_ = VK_NULL_HANDLE;
        vkDestroyDevice(device_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        device_ = VK_NULL_HANDLE;
        instance_ = VK_NULL_HANDLE;
        return SLOTH_VK_ERROR_INIT_FAILED;
    }

    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties_);
    copy_name(device_info_.device_name, out_device_name, max_len);

    initialized_.store(true, std::memory_order_release);
    return SLOTH_VK_SUCCESS;
}

DeviceInfo VulkanContext::describe_device(VkPhysicalDevice device, uint32_t compute_family) const {
    DeviceInfo info;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    info.device_name = props.deviceName;
    info.vendor_id = props.vendorID;
    info.device_id = props.deviceID;
    info.api_version = props.apiVersion;
    info.driver_version = props.driverVersion;
    info.device_type = props.deviceType;
    info.compute_queue_family_index = compute_family;
    std::copy(std::begin(props.limits.maxComputeWorkGroupCount), std::end(props.limits.maxComputeWorkGroupCount),
              std::begin(info.max_compute_work_group_count));

    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(device, &memory);
    info.total_vram_bytes = device_local_bytes(memory);

    // Расширенные свойства: vkGetPhysicalDeviceProperties2 есть с Vulkan 1.1
    if (instance_api_version_ < VK_API_VERSION_1_1 || props.apiVersion < VK_API_VERSION_1_1) {
        return info;
    }
    info.memory_budget_supported = has_device_extension(device, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    const bool driver_props_available = props.apiVersion >= VK_API_VERSION_1_2 ||
        has_device_extension(device, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME);

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup;
    if (driver_props_available) {
        subgroup.pNext = &driver;
    }
    vkGetPhysicalDeviceProperties2(device, &props2);

    info.subgroup_size = subgroup.subgroupSize;
    if (driver_props_available) {
        info.driver_name = driver.driverName;
        info.driver_info = driver.driverInfo;
    }
    return info;
}

int VulkanContext::select_physical_device(int prefer_discrete) {
    uint32_t device_count = 0;
    VkResult res = vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (res != VK_SUCCESS || device_count == 0) {
        std::cerr << "[SlothVulkan] No Vulkan physical devices found." << std::endl;
        return SLOTH_VK_ERROR_NO_DEVICE;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    res = vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
    if (res != VK_SUCCESS && res != VK_INCOMPLETE) {
        report_failure("vkEnumeratePhysicalDevices", res);
        return SLOTH_VK_ERROR_NO_DEVICE;
    }
    devices.resize(device_count);

    const char* override_env = std::getenv(DEVICE_OVERRIDE_ENV);
    const std::string override_value = override_env ? override_env : "";
    const bool override_is_index = !override_value.empty() &&
        std::all_of(override_value.begin(), override_value.end(), [](unsigned char c) { return std::isdigit(c); });

    bool found = false;
    DeviceInfo best_info;
    VkPhysicalDevice best_device = VK_NULL_HANDLE;

    for (uint32_t index = 0; index < devices.size(); ++index) {
        int compute_family = find_compute_queue_family(devices[index]);
        if (compute_family < 0) continue;
        DeviceInfo info = describe_device(devices[index], static_cast<uint32_t>(compute_family));

        if (!override_value.empty()) {
            // Явный выбор пользователя важнее оценки: номер в списке или часть имени без учёта регистра
            const bool matches = override_is_index
                ? std::strtoul(override_value.c_str(), nullptr, 10) == index
                : to_lower(info.device_name).find(to_lower(override_value)) != std::string::npos;
            if (matches) {
                best_info = info;
                best_device = devices[index];
                found = true;
                break;
            }
            continue;
        }

        // Лучше тип устройства (дискретная или встроенная видеокарта), при равенстве — больше VRAM
        const bool better = !found ||
            device_type_rank(info.device_type, prefer_discrete != 0) > device_type_rank(best_info.device_type, prefer_discrete != 0) ||
            (info.device_type == best_info.device_type && info.total_vram_bytes > best_info.total_vram_bytes);
        if (better) {
            best_info = info;
            best_device = devices[index];
            found = true;
        }
    }

    if (!found) {
        if (!override_value.empty()) {
            std::cerr << "[SlothVulkan] " << DEVICE_OVERRIDE_ENV << "=" << override_value
                      << " does not match any compute-capable Vulkan device." << std::endl;
        } else {
            std::cerr << "[SlothVulkan] No compute-capable Vulkan device found." << std::endl;
        }
        return SLOTH_VK_ERROR_NO_DEVICE;
    }

    physical_device_ = best_device;
    compute_queue_family_ = best_info.compute_queue_family_index;
    device_info_ = best_info;
    return SLOTH_VK_SUCCESS;
}

void VulkanContext::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_initialized()) return;
    // Дожидаемся отправки, которая уже идёт, и не пускаем новые
    std::lock_guard<std::mutex> command_lock(command_mutex_);
    initialized_.store(false, std::memory_order_release);

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
}

uint32_t VulkanContext::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const {
    for (uint32_t i = 0; i < mem_properties_.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_properties_.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    // Точного совпадения нет (например, у встроенной видеокарты): любой подходящий тип
    for (uint32_t i = 0; i < mem_properties_.memoryTypeCount; ++i) {
        if (type_filter & (1u << i)) {
            return i;
        }
    }
    return UINT32_MAX;
}

void VulkanContext::get_vram_info(uint64_t* total, uint64_t* used, uint64_t* free) const {
    uint64_t total_vram = device_local_bytes(mem_properties_);
    uint64_t used_vram = allocated_vram_bytes_.load(std::memory_order_relaxed);
    uint64_t free_vram = (total_vram > used_vram) ? (total_vram - used_vram) : 0;

    if (total) *total = total_vram;
    if (used) *used = used_vram;
    if (free) *free = free_vram;
}

int VulkanContext::get_memory_budget(uint64_t* budget, uint64_t* usage) const {
    if (!is_initialized()) return SLOTH_VK_ERROR_NOT_INITIALIZED;
    if (!device_info_.memory_budget_supported) return SLOTH_VK_ERROR_NOT_SUPPORTED;

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_props{};
    budget_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 memory2{};
    memory2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memory2.pNext = &budget_props;
    vkGetPhysicalDeviceMemoryProperties2(physical_device_, &memory2);

    uint64_t total_budget = 0;
    uint64_t total_usage = 0;
    for (uint32_t i = 0; i < memory2.memoryProperties.memoryHeapCount; ++i) {
        if (memory2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total_budget += budget_props.heapBudget[i];
            total_usage += budget_props.heapUsage[i];
        }
    }
    if (budget) *budget = total_budget;
    if (usage) *usage = total_usage;
    return SLOTH_VK_SUCCESS;
}

int VulkanContext::run_commands(const std::function<void(VkCommandBuffer)>& record) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    if (!is_initialized()) return SLOTH_VK_ERROR_NOT_INITIALIZED;

    SubmitResources resources{device_, command_pool_};

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool_;
    alloc_info.commandBufferCount = 1;
    VkResult res = vkAllocateCommandBuffers(device_, &alloc_info, &resources.cmd);
    if (res != VK_SUCCESS) {
        resources.cmd = VK_NULL_HANDLE;
        return report_failure("vkAllocateCommandBuffers", res);
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(resources.cmd, &begin_info);
    if (res != VK_SUCCESS) return report_failure("vkBeginCommandBuffer", res);

    record(resources.cmd);

    res = vkEndCommandBuffer(resources.cmd);
    if (res != VK_SUCCESS) return report_failure("vkEndCommandBuffer", res);

    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    res = vkCreateFence(device_, &fence_ci, nullptr, &resources.fence);
    if (res != VK_SUCCESS) {
        resources.fence = VK_NULL_HANDLE;
        return report_failure("vkCreateFence", res);
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &resources.cmd;
    res = vkQueueSubmit(compute_queue_, 1, &submit_info, resources.fence);
    if (res != VK_SUCCESS) return report_failure("vkQueueSubmit", res);

    // VK_ERROR_DEVICE_LOST (сброс драйвера, перегрев) приходит отсюда
    res = vkWaitForFences(device_, 1, &resources.fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) return report_failure("vkWaitForFences", res);

    return SLOTH_VK_SUCCESS;
}

} // namespace sloth
