#include "sloth_vulkan.h"
#include "vk_context.h"
#include "vk_buffer.h"
#include "vk_pipeline.h"

#include <iostream>
#include <vector>

using namespace sloth;

int sloth_vk_init(int prefer_discrete, char* out_device_name, size_t max_len) {
    return VulkanContext::instance().init(prefer_discrete, out_device_name, max_len);
}

void sloth_vk_shutdown(void) {
    PipelineManager::instance().destroy_all();
    BufferManager::instance().free_all();
    VulkanContext::instance().shutdown();
}

int sloth_vk_get_vram_info(uint64_t* total_bytes, uint64_t* used_bytes, uint64_t* free_bytes) {
    auto& ctx = VulkanContext::instance();
    if (!ctx.is_initialized()) return SLOTH_VK_ERROR_NOT_INITIALIZED;

    ctx.get_vram_info(total_bytes, used_bytes, free_bytes);
    return SLOTH_VK_SUCCESS;
}

SlothBufferHandle sloth_vk_alloc_buffer(size_t size_bytes, int is_device_local) {
    return BufferManager::instance().allocate(size_bytes, is_device_local != 0);
}

void sloth_vk_free_buffer(SlothBufferHandle handle) {
    BufferManager::instance().free_buffer(handle);
}

int sloth_vk_write_buffer(SlothBufferHandle handle, const void* src, size_t size_bytes) {
    return BufferManager::instance().write(handle, src, size_bytes);
}

int sloth_vk_read_buffer(SlothBufferHandle handle, void* dst, size_t size_bytes) {
    return BufferManager::instance().read(handle, dst, size_bytes);
}

int sloth_vk_forward_gemm(
    SlothBufferHandle A,
    SlothBufferHandle B,
    SlothBufferHandle C,
    uint32_t M,
    uint32_t K,
    uint32_t N
) {
    if (M == 0 || K == 0 || N == 0) return SLOTH_VK_SUCCESS;

    auto bufA = BufferManager::instance().get_buffer(A);
    auto bufB = BufferManager::instance().get_buffer(B);
    auto bufC = BufferManager::instance().get_buffer(C);

    if (!bufA || !bufB || !bufC) return SLOTH_VK_ERROR_INVALID_PARAM;

    struct GemmPush {
        uint32_t M;
        uint32_t K;
        uint32_t N;
        uint32_t lda;
        uint32_t ldb;
        uint32_t ldc;
    } pc{M, K, N, K, N, N};

    auto pipeline = PipelineManager::instance().get_pipeline("gemm_f32", 3, sizeof(GemmPush));
    if (!pipeline) return SLOTH_VK_ERROR_SHADER_FAILED;

    uint32_t grid_x = (N + 15) / 16;
    uint32_t grid_y = (M + 15) / 16;

    return pipeline->dispatch(grid_x, grid_y, 1, &pc, sizeof(pc), {bufA, bufB, bufC});
}

int sloth_vk_forward_lora(
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
) {
    uint32_t total_tokens = batch * seq;
    if (total_tokens == 0 || in_dim == 0 || out_dim == 0) return SLOTH_VK_SUCCESS;
    // Шейдер хранит промежуточный вектор в общем массиве фиксированного размера: больший ранг
    // раньше молча обрезался до 128 и давал неверный результат
    if (rank > SLOTH_VK_LORA_MAX_RANK) {
        std::cerr << "[SlothVulkan] LoRA rank " << rank << " exceeds shader limit "
                  << SLOTH_VK_LORA_MAX_RANK << std::endl;
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }

    auto bufX = BufferManager::instance().get_buffer(X);
    auto bufW = BufferManager::instance().get_buffer(W_base);
    auto bufA = BufferManager::instance().get_buffer(A_lora);
    auto bufB = BufferManager::instance().get_buffer(B_lora);
    auto bufOut = BufferManager::instance().get_buffer(Out);

    if (!bufX || !bufA || !bufB || !bufOut) return SLOTH_VK_ERROR_INVALID_PARAM;
    // Нулевой дескриптор W означает «без базового веса», а неверный — ошибку вызывающего
    if (W_base != SLOTH_NULL_BUFFER && !bufW) return SLOTH_VK_ERROR_INVALID_PARAM;

    uint32_t has_base = 1;
    if (!bufW) {
        bufW = bufX; // Provide valid buffer handle for descriptor set binding
        has_base = 0;
    }

    struct LoraPush {
        uint32_t batch_seq;
        uint32_t in_dim;
        uint32_t out_dim;
        uint32_t rank;
        float alpha;
        uint32_t layout_mode; // 0 = PyTorch PEFT
        uint32_t has_base_weight;
    } pc{total_tokens, in_dim, out_dim, rank, alpha, 0, has_base};

    auto pipeline = PipelineManager::instance().get_pipeline("lora_forward", 5, sizeof(LoraPush));
    if (!pipeline) return SLOTH_VK_ERROR_SHADER_FAILED;

    uint32_t grid_x = (out_dim + 15) / 16;
    uint32_t grid_y = (total_tokens + 3) / 4;

    return pipeline->dispatch(grid_x, grid_y, 1, &pc, sizeof(pc), {bufX, bufW, bufA, bufB, bufOut});
}

int sloth_vk_backward_lora(
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
) {
    uint32_t total_tokens = batch * seq;
    if (total_tokens == 0 || in_dim == 0 || out_dim == 0 || rank == 0) return SLOTH_VK_SUCCESS;

    auto bufX = BufferManager::instance().get_buffer(X);
    auto bufdOut = BufferManager::instance().get_buffer(dOut);
    auto bufA = BufferManager::instance().get_buffer(A_lora);
    auto bufB = BufferManager::instance().get_buffer(B_lora);
    auto bufdA = BufferManager::instance().get_buffer(dA_grad);
    auto bufdB = BufferManager::instance().get_buffer(dB_grad);

    if (!bufX || !bufdOut || !bufA || !bufB || !bufdA || !bufdB) return SLOTH_VK_ERROR_INVALID_PARAM;

    struct LoraBackPush {
        uint32_t batch_seq;
        uint32_t in_dim;
        uint32_t out_dim;
        uint32_t rank;
        float alpha;
        uint32_t layout_mode;
        uint32_t pass_mode; // 0 = dB, 1 = dA
    };

    auto pipeline = PipelineManager::instance().get_pipeline("lora_backward", 6, sizeof(LoraBackPush));
    if (!pipeline) return SLOTH_VK_ERROR_SHADER_FAILED;

    std::vector<std::shared_ptr<SlothBuffer>> bufs = {bufX, bufdOut, bufA, bufB, bufdA, bufdB};

    // Pass 0: Compute dB_grad
    LoraBackPush pc_b{total_tokens, in_dim, out_dim, rank, alpha, 0, 0};
    uint32_t total_b = out_dim * rank;
    uint32_t grid_b = (total_b + 63) / 64;
    int res = pipeline->dispatch(grid_b, 1, 1, &pc_b, sizeof(pc_b), bufs);
    if (res != SLOTH_VK_SUCCESS) return res;

    // Pass 1: Compute dA_grad
    LoraBackPush pc_a{total_tokens, in_dim, out_dim, rank, alpha, 0, 1};
    uint32_t total_a = rank * in_dim;
    uint32_t grid_a = (total_a + 63) / 64;
    res = pipeline->dispatch(grid_a, 1, 1, &pc_a, sizeof(pc_a), bufs);

    return res;
}

int sloth_vk_adamw_step(
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
) {
    if (num_elements == 0) return SLOTH_VK_SUCCESS;

    auto bufW = BufferManager::instance().get_buffer(Weights);
    auto bufG = BufferManager::instance().get_buffer(Grads);
    auto bufM = BufferManager::instance().get_buffer(M_state);
    auto bufV = BufferManager::instance().get_buffer(V_state);

    if (!bufW || !bufG || !bufM || !bufV) return SLOTH_VK_ERROR_INVALID_PARAM;

    struct AdamPush {
        uint32_t num_elements;
        float lr;
        float beta1;
        float beta2;
        float eps;
        float weight_decay;
        uint32_t step;
    } pc{num_elements, lr, beta1, beta2, eps, weight_decay, step};

    auto pipeline = PipelineManager::instance().get_pipeline("adamw", 4, sizeof(AdamPush));
    if (!pipeline) return SLOTH_VK_ERROR_SHADER_FAILED;

    uint32_t grid_x = (num_elements + 63) / 64;
    return pipeline->dispatch(grid_x, 1, 1, &pc, sizeof(pc), {bufW, bufG, bufM, bufV});
}

int sloth_vk_rmsnorm_forward(
    SlothBufferHandle X,
    SlothBufferHandle Gamma,
    SlothBufferHandle Out,
    uint32_t batch_seq,
    uint32_t dim,
    float eps
) {
    if (batch_seq == 0 || dim == 0) return SLOTH_VK_SUCCESS;

    auto bufX = BufferManager::instance().get_buffer(X);
    auto bufGamma = BufferManager::instance().get_buffer(Gamma);
    auto bufOut = BufferManager::instance().get_buffer(Out);

    if (!bufX || !bufGamma || !bufOut) return SLOTH_VK_ERROR_INVALID_PARAM;

    struct RmsPush {
        uint32_t batch_seq;
        uint32_t dim;
        float eps;
        uint32_t mode; // 0 = forward
    } pc{batch_seq, dim, eps, 0};

    auto pipeline = PipelineManager::instance().get_pipeline("rmsnorm", 6, sizeof(RmsPush));
    if (!pipeline) return SLOTH_VK_ERROR_SHADER_FAILED;

    return pipeline->dispatch(batch_seq, 1, 1, &pc, sizeof(pc), {bufX, bufGamma, bufOut, bufOut, bufOut, bufOut});
}

int sloth_vk_rmsnorm_backward(
    SlothBufferHandle dOut,
    SlothBufferHandle X,
    SlothBufferHandle Gamma,
    SlothBufferHandle dX,
    SlothBufferHandle dGamma,
    uint32_t batch_seq,
    uint32_t dim,
    float eps
) {
    if (batch_seq == 0 || dim == 0) return SLOTH_VK_SUCCESS;

    auto bufdOut = BufferManager::instance().get_buffer(dOut);
    auto bufX = BufferManager::instance().get_buffer(X);
    auto bufGamma = BufferManager::instance().get_buffer(Gamma);
    auto bufdX = BufferManager::instance().get_buffer(dX);
    auto bufdGamma = BufferManager::instance().get_buffer(dGamma);

    if (!bufdOut || !bufX || !bufGamma || !bufdX || !bufdGamma) return SLOTH_VK_ERROR_INVALID_PARAM;

    struct RmsPush {
        uint32_t batch_seq;
        uint32_t dim;
        float eps;
        uint32_t mode;
    };

    auto pipeline = PipelineManager::instance().get_pipeline("rmsnorm", 6, sizeof(RmsPush));
    if (!pipeline) return SLOTH_VK_ERROR_SHADER_FAILED;

    std::vector<std::shared_ptr<SlothBuffer>> bufs = {bufX, bufGamma, bufdX, bufdOut, bufdX, bufdGamma};

    // Pass 1: compute dX
    RmsPush pc_dx{batch_seq, dim, eps, 1};
    int res = pipeline->dispatch(batch_seq, 1, 1, &pc_dx, sizeof(pc_dx), bufs);
    if (res != SLOTH_VK_SUCCESS) return res;

    // Pass 2: compute dGamma
    RmsPush pc_dg{batch_seq, dim, eps, 2};
    uint32_t grid_dim = (dim + 63) / 64;
    return pipeline->dispatch(grid_dim, 1, 1, &pc_dg, sizeof(pc_dg), bufs);
}
