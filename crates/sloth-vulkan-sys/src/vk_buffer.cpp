#include "vk_buffer.h"
#include "vk_context.h"
#include "sloth_vulkan.h"

#include <iostream>
#include <cstring>

namespace sloth {

BufferManager& BufferManager::instance() {
    static BufferManager mgr;
    return mgr;
}

BufferManager::~BufferManager() {
    free_all();
}

void BufferManager::free_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : buffers_) {
        destroy_internal_buffer(pair.second.get());
    }
    buffers_.clear();
}

std::shared_ptr<SlothBuffer> BufferManager::create_internal_buffer(size_t size_bytes, bool is_device_local) {
    auto& ctx = VulkanContext::instance();
    if (!ctx.is_initialized()) {
        std::cerr << "[SlothVulkan] Context not initialized before buffer allocation." << std::endl;
        return nullptr;
    }

    VkDevice device = ctx.get_device();
    size_t actual_size = (size_bytes > 0) ? size_bytes : 16;

    auto buf = std::make_shared<SlothBuffer>();
    buf->size = size_bytes;
    buf->is_device_local = is_device_local;

    VkBufferCreateInfo buffer_ci{};
    buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_ci.size = actual_size;
    buffer_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = vkCreateBuffer(device, &buffer_ci, nullptr, &buf->buffer);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to create VkBuffer: " << res << std::endl;
        return nullptr;
    }

    VkMemoryRequirements mem_reqs{};
    vkGetBufferMemoryRequirements(device, buf->buffer, &mem_reqs);
    buf->allocation_size = mem_reqs.size;

    VkMemoryPropertyFlags prop_flags = 0;
    if (is_device_local) {
        prop_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else {
        prop_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    uint32_t mem_type_idx = ctx.find_memory_type(mem_reqs.memoryTypeBits, prop_flags);
    if (mem_type_idx == UINT32_MAX) {
        std::cerr << "[SlothVulkan] Failed to find suitable memory type for buffer." << std::endl;
        vkDestroyBuffer(device, buf->buffer, nullptr);
        return nullptr;
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = mem_type_idx;

    res = vkAllocateMemory(device, &alloc_info, nullptr, &buf->memory);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to allocate device memory: " << res << std::endl;
        vkDestroyBuffer(device, buf->buffer, nullptr);
        return nullptr;
    }

    res = vkBindBufferMemory(device, buf->buffer, buf->memory, 0);
    if (res != VK_SUCCESS) {
        std::cerr << "[SlothVulkan] Failed to bind buffer memory: " << res << std::endl;
        vkFreeMemory(device, buf->memory, nullptr);
        vkDestroyBuffer(device, buf->buffer, nullptr);
        return nullptr;
    }

    if (!is_device_local) {
        res = vkMapMemory(device, buf->memory, 0, mem_reqs.size, 0, &buf->mapped_ptr);
        if (res != VK_SUCCESS) {
            std::cerr << "[SlothVulkan] Failed to map host buffer memory: " << res << std::endl;
            vkFreeMemory(device, buf->memory, nullptr);
            vkDestroyBuffer(device, buf->buffer, nullptr);
            return nullptr;
        }
    }

    if (is_device_local) {
        ctx.add_allocated_vram(buf->allocation_size);
    }

    return buf;
}

void BufferManager::destroy_internal_buffer(SlothBuffer* buf) {
    if (!buf) return;
    auto& ctx = VulkanContext::instance();
    if (!ctx.is_initialized()) return;

    VkDevice device = ctx.get_device();

    if (buf->is_device_local) {
        ctx.sub_allocated_vram(buf->allocation_size);
    }

    if (buf->mapped_ptr && buf->memory != VK_NULL_HANDLE) {
        vkUnmapMemory(device, buf->memory);
        buf->mapped_ptr = nullptr;
    }

    if (buf->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buf->buffer, nullptr);
        buf->buffer = VK_NULL_HANDLE;
    }

    if (buf->memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buf->memory, nullptr);
        buf->memory = VK_NULL_HANDLE;
    }
}

SlothBufferHandle BufferManager::allocate(size_t size_bytes, bool is_device_local) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto buf = create_internal_buffer(size_bytes, is_device_local);
    if (!buf) return SLOTH_NULL_BUFFER;

    SlothBufferHandle handle = next_handle_++;
    buf->handle = handle;
    buffers_[handle] = buf;
    return handle;
}

void BufferManager::free_buffer(SlothBufferHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(handle);
    if (it != buffers_.end()) {
        destroy_internal_buffer(it->second.get());
        buffers_.erase(it);
    }
}

std::shared_ptr<SlothBuffer> BufferManager::get_buffer(SlothBufferHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(handle);
    if (it != buffers_.end()) {
        return it->second;
    }
    return nullptr;
}

int BufferManager::write(SlothBufferHandle handle, const void* src, size_t size_bytes) {
    if (!src || size_bytes == 0) return SLOTH_VK_SUCCESS;

    std::shared_ptr<SlothBuffer> buf = get_buffer(handle);
    if (!buf) {
        std::cerr << "[SlothVulkan] Write to invalid buffer handle: " << handle << std::endl;
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }

    if (size_bytes > buf->size) {
        std::cerr << "[SlothVulkan] Buffer write size exceeds buffer capacity." << std::endl;
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }

    if (!buf->is_device_local) {
        // Direct host copy
        if (!buf->mapped_ptr) return SLOTH_VK_ERROR_INVALID_PARAM;
        std::memcpy(buf->mapped_ptr, src, size_bytes);
        return SLOTH_VK_SUCCESS;
    }

    // Staging buffer copy for Device-Local VRAM
    auto staging = create_internal_buffer(size_bytes, false);
    if (!staging || !staging->mapped_ptr) {
        return SLOTH_VK_ERROR_OUT_OF_MEMORY;
    }

    std::memcpy(staging->mapped_ptr, src, size_bytes);

    auto& ctx = VulkanContext::instance();
    VkCommandBuffer cmd = ctx.begin_single_time_commands();

    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size_bytes;

    vkCmdCopyBuffer(cmd, staging->buffer, buf->buffer, 1, &copy_region);

    // Memory barrier to make transfer write visible to shader read
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buf->buffer;
    barrier.offset = 0;
    barrier.size = size_bytes;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr
    );

    ctx.end_single_time_commands(cmd);
    destroy_internal_buffer(staging.get());

    return SLOTH_VK_SUCCESS;
}

int BufferManager::read(SlothBufferHandle handle, void* dst, size_t size_bytes) {
    if (!dst || size_bytes == 0) return SLOTH_VK_SUCCESS;

    std::shared_ptr<SlothBuffer> buf = get_buffer(handle);
    if (!buf) {
        std::cerr << "[SlothVulkan] Read from invalid buffer handle: " << handle << std::endl;
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }

    if (size_bytes > buf->size) {
        std::cerr << "[SlothVulkan] Buffer read size exceeds buffer capacity." << std::endl;
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }

    if (!buf->is_device_local) {
        if (!buf->mapped_ptr) return SLOTH_VK_ERROR_INVALID_PARAM;
        std::memcpy(dst, buf->mapped_ptr, size_bytes);
        return SLOTH_VK_SUCCESS;
    }

    // Staging buffer copy from Device-Local VRAM
    auto staging = create_internal_buffer(size_bytes, false);
    if (!staging || !staging->mapped_ptr) {
        return SLOTH_VK_ERROR_OUT_OF_MEMORY;
    }

    auto& ctx = VulkanContext::instance();
    VkCommandBuffer cmd = ctx.begin_single_time_commands();

    // Barrier ensuring preceding compute writes are visible to transfer read
    VkBufferMemoryBarrier pre_barrier{};
    pre_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    pre_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    pre_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_barrier.buffer = buf->buffer;
    pre_barrier.offset = 0;
    pre_barrier.size = size_bytes;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        1, &pre_barrier,
        0, nullptr
    );

    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size_bytes;

    vkCmdCopyBuffer(cmd, buf->buffer, staging->buffer, 1, &copy_region);

    ctx.end_single_time_commands(cmd);

    std::memcpy(dst, staging->mapped_ptr, size_bytes);
    destroy_internal_buffer(staging.get());

    return SLOTH_VK_SUCCESS;
}

} // namespace sloth
