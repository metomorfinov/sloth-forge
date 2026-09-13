#ifndef SLOTH_VULKAN_H
#define SLOTH_VULKAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SLOTH_VK_STATIC: библиотека встраивается в программу статически (так её собирает build.rs),
   поэтому на Windows не нужны dllimport/dllexport. */
#if defined(SLOTH_VK_STATIC)
  #define SLOTH_VK_API
#elif defined(_WIN32) || defined(__CYGWIN__)
  #if defined(SLOTH_VK_BUILD_SHARED)
    #define SLOTH_VK_API __declspec(dllexport)
  #else
    #define SLOTH_VK_API __declspec(dllimport)
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define SLOTH_VK_API __attribute__((visibility("default")))
  #else
    #define SLOTH_VK_API
  #endif
#endif

/* Status / Error Codes */
#define SLOTH_VK_SUCCESS                0
#define SLOTH_VK_ERROR_INIT_FAILED     -1
#define SLOTH_VK_ERROR_NO_DEVICE       -2
#define SLOTH_VK_ERROR_OUT_OF_MEMORY   -3
#define SLOTH_VK_ERROR_INVALID_PARAM   -4
#define SLOTH_VK_ERROR_SHADER_FAILED   -5
#define SLOTH_VK_ERROR_DISPATCH_FAILED -6
#define SLOTH_VK_ERROR_NOT_INITIALIZED -7

/* Handle types */
typedef uint64_t SlothBufferHandle;
#define SLOTH_NULL_BUFFER 0ULL

/**
 * @brief Initialize Vulkan compute instance, physical device, compute queue and memory system.
 * @param prefer_discrete 1 to prefer discrete GPU (e.g. AMD Radeon RX 570), 0 otherwise.
 * @param out_device_name Buffer to receive selected GPU name string (optional, can be NULL).
 * @param max_len Maximum length of out_device_name buffer.
 * @return 0 on success, negative error code otherwise.
 */
SLOTH_VK_API int sloth_vk_init(int prefer_discrete, char* out_device_name, size_t max_len);

/**
 * @brief Free all pipelines, memory pools, command buffers, devices, and destroy Vulkan instance.
 */
SLOTH_VK_API void sloth_vk_shutdown(void);

/**
 * @brief Query total, currently allocated, and free VRAM bytes on active GPU device heap.
 * @param total_bytes Receives total VRAM size in bytes.
 * @param used_bytes Receives currently tracked used VRAM in bytes.
 * @param free_bytes Receives estimated free VRAM in bytes.
 * @return 0 on success, negative error code otherwise.
 */
SLOTH_VK_API int sloth_vk_get_vram_info(uint64_t* total_bytes, uint64_t* used_bytes, uint64_t* free_bytes);

/**
 * @brief Allocate a GPU storage buffer.
 * @param size_bytes Size in bytes.
 * @param is_device_local 1 for high-bandwidth device-local VRAM, 0 for host-visible staging memory.
 * @return SlothBufferHandle or SLOTH_NULL_BUFFER on failure.
 */
SLOTH_VK_API SlothBufferHandle sloth_vk_alloc_buffer(size_t size_bytes, int is_device_local);

/**
 * @brief Free an allocated GPU buffer and its device memory.
 * @param handle Handle to buffer.
 */
SLOTH_VK_API void sloth_vk_free_buffer(SlothBufferHandle handle);

/**
 * @brief Write data from host RAM to GPU buffer (device-local via staging if necessary).
 * @param handle Target GPU buffer handle.
 * @param src Pointer to host source memory.
 * @param size_bytes Number of bytes to copy.
 * @return 0 on success, negative error code otherwise.
 */
SLOTH_VK_API int sloth_vk_write_buffer(SlothBufferHandle handle, const void* src, size_t size_bytes);

/**
 * @brief Read data from GPU buffer to host RAM (device-local via staging if necessary).
 * @param handle Source GPU buffer handle.
 * @param dst Pointer to host destination memory.
 * @param size_bytes Number of bytes to copy.
 * @return 0 on success, negative error code otherwise.
 */
SLOTH_VK_API int sloth_vk_read_buffer(SlothBufferHandle handle, void* dst, size_t size_bytes);

/**
 * @brief Tiled GEMM: C = A x B.
 * A is (M x K), B is (K x N), C is (M x N). All float32, row-major.
 */
SLOTH_VK_API int sloth_vk_forward_gemm(
    SlothBufferHandle A,
    SlothBufferHandle B,
    SlothBufferHandle C,
    uint32_t M,
    uint32_t K,
    uint32_t N
);

/**
 * @brief Fused forward LoRA calculation: Out = X * W_base + (alpha / rank) * (X * A_lora) * B_lora
 *
 * Matrix shapes:
 * X: (batch * seq, in_dim)
 * W_base: (in_dim, out_dim) or (out_dim, in_dim) based on config
 * A_lora: (in_dim, rank) or (rank, in_dim)
 * B_lora: (rank, out_dim) or (out_dim, rank)
 * Out: (batch * seq, out_dim)
 */
SLOTH_VK_API int sloth_vk_forward_lora(
    SlothBufferHandle X,
    SlothBufferHandle W_base,
    SlothBufferHandle A_lora,
    SlothBufferHandle B_lora,
    SlothBufferHandle Out,
    uint32_t batch,
    uint32_t seq,
    uint32_t in_dim,
    uint32_t out_dim,
    uint32_t rank,
    float alpha
);

/**
 * @brief LoRA backward calculating dA and dB gradients directly in VRAM.
 *
 * Computes:
 * intermediate h = X * A_lora (or A * x)
 * dB_grad = (alpha / rank) * (h^T * dOut)
 * dh = (alpha / rank) * (dOut * B_lora^T)
 * dA_grad = X^T * dh
 */
SLOTH_VK_API int sloth_vk_backward_lora(
    SlothBufferHandle X,
    SlothBufferHandle dOut,
    SlothBufferHandle A_lora,
    SlothBufferHandle B_lora,
    SlothBufferHandle dA_grad,
    SlothBufferHandle dB_grad,
    uint32_t batch,
    uint32_t seq,
    uint32_t in_dim,
    uint32_t out_dim,
    uint32_t rank,
    float alpha
);

/**
 * @brief In-place AdamW optimizer step directly on GPU buffers.
 *
 * Weights = Weights - lr * weight_decay * Weights
 * M_state = beta1 * M_state + (1 - beta1) * Grads
 * V_state = beta2 * V_state + (1 - beta2) * (Grads^2)
 * Weights = Weights - lr * (M_state / (1 - beta1^step)) / (sqrt(V_state / (1 - beta2^step)) + eps)
 */
SLOTH_VK_API int sloth_vk_adamw_step(
    SlothBufferHandle Weights,
    SlothBufferHandle Grads,
    SlothBufferHandle M_state,
    SlothBufferHandle V_state,
    uint32_t num_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    uint32_t step
);

/**
 * @brief RMSNorm forward: Out = (X / RMS(X)) * Gamma
 */
SLOTH_VK_API int sloth_vk_rmsnorm_forward(
    SlothBufferHandle X,
    SlothBufferHandle Gamma,
    SlothBufferHandle Out,
    uint32_t batch_seq,
    uint32_t dim,
    float eps
);

/**
 * @brief RMSNorm backward: computes dX and dGamma
 */
SLOTH_VK_API int sloth_vk_rmsnorm_backward(
    SlothBufferHandle dOut,
    SlothBufferHandle X,
    SlothBufferHandle Gamma,
    SlothBufferHandle dX,
    SlothBufferHandle dGamma,
    uint32_t batch_seq,
    uint32_t dim,
    float eps
);

#ifdef __cplusplus
}
#endif

#endif /* SLOTH_VULKAN_H */
