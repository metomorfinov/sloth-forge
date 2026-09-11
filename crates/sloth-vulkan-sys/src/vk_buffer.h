#ifndef SLOTH_VK_BUFFER_H
#define SLOTH_VK_BUFFER_H

#include <vulkan/vulkan.h>
#include "sloth_vulkan.h"

#include <unordered_map>
#include <mutex>
#include <memory>

namespace sloth {

struct SlothBuffer {
    SlothBufferHandle handle = SLOTH_NULL_BUFFER;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t size = 0;
    bool is_device_local = false;
    void* mapped_ptr = nullptr;
    VkDeviceSize allocation_size = 0;
};

class BufferManager {
public:
    static BufferManager& instance();

    SlothBufferHandle allocate(size_t size_bytes, bool is_device_local);
    void free_buffer(SlothBufferHandle handle);
    std::shared_ptr<SlothBuffer> get_buffer(SlothBufferHandle handle);

    int write(SlothBufferHandle handle, const void* src, size_t size_bytes);
    int read(SlothBufferHandle handle, void* dst, size_t size_bytes);

    void free_all();

private:
    BufferManager() = default;
    ~BufferManager();
    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;

    std::shared_ptr<SlothBuffer> create_internal_buffer(size_t size_bytes, bool is_device_local);
    void destroy_internal_buffer(SlothBuffer* buf);

    std::mutex mutex_;
    SlothBufferHandle next_handle_ = 1;
    std::unordered_map<SlothBufferHandle, std::shared_ptr<SlothBuffer>> buffers_;
};

} // namespace sloth

#endif // SLOTH_VK_BUFFER_H
