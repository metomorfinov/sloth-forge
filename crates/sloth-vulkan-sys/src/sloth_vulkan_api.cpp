#include "sloth_vulkan.h"
#include "vk_context.h"
#include "vk_buffer.h"
#include "vk_pipeline.h"

#include <cstring>
#include <iostream>
#include <vector>

using namespace sloth;

namespace {

using BufferPtr = std::shared_ptr<SlothBuffer>;

// Шейдеры адресуют элементы 32-битным uint: больше элементов в буфере не адресовать
const uint64_t MAX_SHADER_ELEMENTS = UINT32_MAX;
const uint64_t FLOAT_BYTES = sizeof(float);

// Размеры рабочих групп шейдеров (layout local_size и шаг тайла GEMM)
const uint32_t GEMM_TILE = 16;
const uint32_t LORA_FORWARD_CHANNELS_PER_GROUP = 16;
const uint32_t LORA_FORWARD_TOKENS_PER_GROUP = 4;
const uint32_t LINEAR_GROUP = 64;

BufferPtr lookup(SlothBufferHandle handle) {
    return BufferManager::instance().get_buffer(handle);
}

int invalid(const char* operation, const char* reason) {
    std::cerr << "[SlothVulkan] " << operation << ": " << reason << std::endl;
    return SLOTH_VK_ERROR_INVALID_PARAM;
}

// Буфер существует и вмещает count чисел float32. Раньше размеры не проверялись и шейдер
// мог читать и писать за концом буфера.
bool holds_floats(const BufferPtr& buf, uint64_t count) {
    return buf && count <= MAX_SHADER_ELEMENTS && buf->size / FLOAT_BYTES >= count;
}

// ceil(total / per_group) в uint32; false — не помещается.
bool group_count(uint64_t total, uint32_t per_group, uint32_t* out) {
    uint64_t groups = (total + per_group - 1) / per_group;
    if (groups > UINT32_MAX) return false;
    *out = static_cast<uint32_t>(groups);
    return true;
}

void copy_string(const std::string& text, char* out, size_t size) {
    std::strncpy(out, text.c_str(), size - 1);
    out[size - 1] = '\0';
}

} // namespace

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

int sloth_vk_get_device_info(SlothDeviceInfo* out) {
    if (!out) return SLOTH_VK_ERROR_INVALID_PARAM;
    auto& ctx = VulkanContext::instance();
    if (!ctx.is_initialized()) return SLOTH_VK_ERROR_NOT_INITIALIZED;

    const DeviceInfo& info = ctx.get_device_info();
    std::memset(out, 0, sizeof(*out));
    copy_string(info.device_name, out->device_name, sizeof(out->device_name));
    copy_string(info.driver_name, out->driver_name, sizeof(out->driver_name));
    copy_string(info.driver_info, out->driver_info, sizeof(out->driver_info));
    out->vendor_id = info.vendor_id;
    out->device_id = info.device_id;
    out->api_version = info.api_version;
    out->driver_version = info.driver_version;
    out->device_type = static_cast<uint32_t>(info.device_type);
    out->subgroup_size = info.subgroup_size;
    for (size_t axis = 0; axis < 3; ++axis) {
        out->max_compute_work_group_count[axis] = info.max_compute_work_group_count[axis];
    }
    out->memory_budget_supported = info.memory_budget_supported ? 1 : 0;
    out->device_local_bytes = info.total_vram_bytes;
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_get_memory_budget(uint64_t* budget_bytes, uint64_t* usage_bytes) {
    if (!budget_bytes || !usage_bytes) return SLOTH_VK_ERROR_INVALID_PARAM;
    return VulkanContext::instance().get_memory_budget(budget_bytes, usage_bytes);
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
    const char* op = "forward_gemm";
    if (M == 0 || K == 0 || N == 0) return SLOTH_VK_SUCCESS;

    auto bufA = lookup(A);
    auto bufB = lookup(B);
    auto bufC = lookup(C);
    if (!holds_floats(bufA, uint64_t{M} * K) || !holds_floats(bufB, uint64_t{K} * N) ||
        !holds_floats(bufC, uint64_t{M} * N)) {
        return invalid(op, "buffer missing or smaller than M*K, K*N, M*N");
    }

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

    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    group_count(N, GEMM_TILE, &grid_x);
    group_count(M, GEMM_TILE, &grid_y);
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
    const char* op = "forward_lora";
    const uint64_t total_tokens = uint64_t{batch} * seq;
    if (total_tokens == 0 || in_dim == 0 || out_dim == 0) return SLOTH_VK_SUCCESS;
    if (total_tokens > UINT32_MAX) return invalid(op, "batch * seq does not fit in uint32");
    // Шейдер хранит промежуточный вектор в общем массиве фиксированного размера: больший ранг
    // раньше молча обрезался до 128 и давал неверный результат
    if (rank > SLOTH_VK_LORA_MAX_RANK) {
        std::cerr << "[SlothVulkan] LoRA rank " << rank << " exceeds shader limit "
                  << SLOTH_VK_LORA_MAX_RANK << std::endl;
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }

    auto bufX = lookup(X);
    auto bufW = lookup(W_base);
    auto bufA = lookup(A_lora);
    auto bufB = lookup(B_lora);
    auto bufOut = lookup(Out);

    if (!holds_floats(bufX, total_tokens * in_dim) || !holds_floats(bufA, uint64_t{rank} * in_dim) ||
        !holds_floats(bufB, uint64_t{out_dim} * rank) || !holds_floats(bufOut, total_tokens * out_dim)) {
        return invalid(op, "buffer missing or smaller than its LoRA shape");
    }
    // Нулевой дескриптор W означает «без базового веса», а неверный — ошибку вызывающего
    if (W_base != SLOTH_NULL_BUFFER && !holds_floats(bufW, uint64_t{out_dim} * in_dim)) {
        return invalid(op, "base weight buffer missing or smaller than out_dim*in_dim");
    }

    uint32_t has_base = 1;
    if (!bufW) {
        bufW = bufX; // Привязке дескриптора нужен существующий буфер
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
    } pc{static_cast<uint32_t>(total_tokens), in_dim, out_dim, rank, alpha, 0, has_base};

    auto pipeline = PipelineManager::instance().get_pipeline("lora_forward", 5, sizeof(LoraPush));
    if (!pipeline) return SLOTH_VK_ERROR_SHADER_FAILED;

    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    group_count(out_dim, LORA_FORWARD_CHANNELS_PER_GROUP, &grid_x);
    group_count(total_tokens, LORA_FORWARD_TOKENS_PER_GROUP, &grid_y);
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
    const char* op = "backward_lora";
    const uint64_t total_tokens = uint64_t{batch} * seq;
    if (total_tokens == 0 || in_dim == 0 || out_dim == 0 || rank == 0) return SLOTH_VK_SUCCESS;
    if (total_tokens > UINT32_MAX) return invalid(op, "batch * seq does not fit in uint32");

    auto bufX = lookup(X);
    auto bufdOut = lookup(dOut);
    auto bufA = lookup(A_lora);
    auto bufB = lookup(B_lora);
    auto bufdA = lookup(dA_grad);
    auto bufdB = lookup(dB_grad);

    const uint64_t a_elements = uint64_t{rank} * in_dim;
    const uint64_t b_elements = uint64_t{out_dim} * rank;
    if (!holds_floats(bufX, total_tokens * in_dim) || !holds_floats(bufdOut, total_tokens * out_dim) ||
        !holds_floats(bufA, a_elements) || !holds_floats(bufB, b_elements) ||
        !holds_floats(bufdA, a_elements) || !holds_floats(bufdB, b_elements)) {
        return invalid(op, "buffer missing or smaller than its LoRA shape");
    }

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
    const auto batch_seq = static_cast<uint32_t>(total_tokens);

    // Проход 0: dB
    LoraBackPush pc_b{batch_seq, in_dim, out_dim, rank, alpha, 0, 0};
    uint32_t grid_b = 0;
    group_count(b_elements, LINEAR_GROUP, &grid_b);
    int res = pipeline->dispatch(grid_b, 1, 1, &pc_b, sizeof(pc_b), bufs);
    if (res != SLOTH_VK_SUCCESS) return res;

    // Проход 1: dA
    LoraBackPush pc_a{batch_seq, in_dim, out_dim, rank, alpha, 0, 1};
    uint32_t grid_a = 0;
    group_count(a_elements, LINEAR_GROUP, &grid_a);
    return pipeline->dispatch(grid_a, 1, 1, &pc_a, sizeof(pc_a), bufs);
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

    auto bufW = lookup(Weights);
    auto bufG = lookup(Grads);
    auto bufM = lookup(M_state);
    auto bufV = lookup(V_state);
    if (!holds_floats(bufW, num_elements) || !holds_floats(bufG, num_elements) ||
        !holds_floats(bufM, num_elements) || !holds_floats(bufV, num_elements)) {
        return invalid("adamw_step", "buffer missing or smaller than num_elements");
    }

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

    uint32_t grid_x = 0;
    group_count(num_elements, LINEAR_GROUP, &grid_x);
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

    auto bufX = lookup(X);
    auto bufGamma = lookup(Gamma);
    auto bufOut = lookup(Out);
    const uint64_t elements = uint64_t{batch_seq} * dim;
    if (!holds_floats(bufX, elements) || !holds_floats(bufGamma, dim) || !holds_floats(bufOut, elements)) {
        return invalid("rmsnorm_forward", "buffer missing or smaller than batch_seq*dim, dim");
    }

    struct RmsPush {
        uint32_t batch_seq;
        uint32_t dim;
        float eps;
        uint32_t mode; // 0 = forward
    } pc{batch_seq, dim, eps, 0};

    auto pipeline = PipelineManager::instance().get_pipeline("rmsnorm", 6, sizeof(RmsPush));
    if (!pipeline) return SLOTH_VK_ERROR_SHADER_FAILED;

    // Одна рабочая группа на строку: batch_seq ограничен лимитом устройства (проверяет dispatch)
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

    auto bufdOut = lookup(dOut);
    auto bufX = lookup(X);
    auto bufGamma = lookup(Gamma);
    auto bufdX = lookup(dX);
    auto bufdGamma = lookup(dGamma);
    const uint64_t elements = uint64_t{batch_seq} * dim;
    if (!holds_floats(bufdOut, elements) || !holds_floats(bufX, elements) || !holds_floats(bufGamma, dim) ||
        !holds_floats(bufdX, elements) || !holds_floats(bufdGamma, dim)) {
        return invalid("rmsnorm_backward", "buffer missing or smaller than batch_seq*dim, dim");
    }

    struct RmsPush {
        uint32_t batch_seq;
        uint32_t dim;
        float eps;
        uint32_t mode;
    };

    auto pipeline = PipelineManager::instance().get_pipeline("rmsnorm", 6, sizeof(RmsPush));
    if (!pipeline) return SLOTH_VK_ERROR_SHADER_FAILED;

    std::vector<std::shared_ptr<SlothBuffer>> bufs = {bufX, bufGamma, bufdX, bufdOut, bufdX, bufdGamma};

    // Проход 1: dX (одна группа на строку)
    RmsPush pc_dx{batch_seq, dim, eps, 1};
    int res = pipeline->dispatch(batch_seq, 1, 1, &pc_dx, sizeof(pc_dx), bufs);
    if (res != SLOTH_VK_SUCCESS) return res;

    // Проход 2: dGamma
    RmsPush pc_dg{batch_seq, dim, eps, 2};
    uint32_t grid_dim = 0;
    group_count(dim, LINEAR_GROUP, &grid_dim);
    return pipeline->dispatch(grid_dim, 1, 1, &pc_dg, sizeof(pc_dg), bufs);
}
