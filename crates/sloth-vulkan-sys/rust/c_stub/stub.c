#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "../../include/sloth_vulkan.h"

#define MAX_BUFFERS 4096

typedef struct {
    void* ptr;
    size_t size;
    int is_device_local;
    int in_use;
} BufferEntry;

static BufferEntry g_buffers[MAX_BUFFERS] = {0};
static uint64_t g_used_vram = 0;
static const uint64_t TOTAL_VRAM = 4ULL * 1024ULL * 1024ULL * 1024ULL; // 4 GiB (RX 570)
static int g_initialized = 0;

int sloth_vk_init(int prefer_discrete, char* out_device_name, size_t max_len) {
    (void)prefer_discrete;
    g_initialized = 1;
    if (out_device_name && max_len > 0) {
        /* Заглушка без Vulkan: не выдаём себя за настоящую видеокарту */
        const char* name = "SlothForge CPU fallback (no Vulkan)";
        strncpy(out_device_name, name, max_len - 1);
        out_device_name[max_len - 1] = '\0';
    }
    return SLOTH_VK_SUCCESS;
}

void sloth_vk_shutdown(void) {
    for (size_t i = 1; i < MAX_BUFFERS; ++i) {
        if (g_buffers[i].in_use && g_buffers[i].ptr) {
            free(g_buffers[i].ptr);
            g_buffers[i].ptr = NULL;
            g_buffers[i].in_use = 0;
        }
    }
    g_used_vram = 0;
    g_initialized = 0;
}

int sloth_vk_get_vram_info(uint64_t* total_bytes, uint64_t* used_bytes, uint64_t* free_bytes) {
    if (total_bytes) *total_bytes = TOTAL_VRAM;
    if (used_bytes) *used_bytes = g_used_vram;
    if (free_bytes) *free_bytes = (TOTAL_VRAM > g_used_vram) ? (TOTAL_VRAM - g_used_vram) : 0;
    return SLOTH_VK_SUCCESS;
}

SlothBufferHandle sloth_vk_alloc_buffer(size_t size_bytes, int is_device_local) {
    for (size_t i = 1; i < MAX_BUFFERS; ++i) {
        if (!g_buffers[i].in_use) {
            void* p = calloc(1, size_bytes);
            if (!p) return SLOTH_NULL_BUFFER;
            g_buffers[i].ptr = p;
            g_buffers[i].size = size_bytes;
            g_buffers[i].is_device_local = is_device_local;
            g_buffers[i].in_use = 1;
            g_used_vram += size_bytes;
            return (SlothBufferHandle)i;
        }
    }
    return SLOTH_NULL_BUFFER;
}

void sloth_vk_free_buffer(SlothBufferHandle handle) {
    if (handle > 0 && handle < MAX_BUFFERS && g_buffers[handle].in_use) {
        if (g_used_vram >= g_buffers[handle].size) {
            g_used_vram -= g_buffers[handle].size;
        } else {
            g_used_vram = 0;
        }
        free(g_buffers[handle].ptr);
        g_buffers[handle].ptr = NULL;
        g_buffers[handle].size = 0;
        g_buffers[handle].in_use = 0;
    }
}

int sloth_vk_write_buffer(SlothBufferHandle handle, const void* src, size_t size_bytes) {
    if (handle == 0 || handle >= MAX_BUFFERS || !g_buffers[handle].in_use || !src) {
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }
    size_t to_copy = (size_bytes <= g_buffers[handle].size) ? size_bytes : g_buffers[handle].size;
    memcpy(g_buffers[handle].ptr, src, to_copy);
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_read_buffer(SlothBufferHandle handle, void* dst, size_t size_bytes) {
    if (handle == 0 || handle >= MAX_BUFFERS || !g_buffers[handle].in_use || !dst) {
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }
    size_t to_copy = (size_bytes <= g_buffers[handle].size) ? size_bytes : g_buffers[handle].size;
    memcpy(dst, g_buffers[handle].ptr, to_copy);
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_forward_gemm(SlothBufferHandle A, SlothBufferHandle B, SlothBufferHandle C, uint32_t M, uint32_t K, uint32_t N) {
    if (A == 0 || B == 0 || C == 0 || A >= MAX_BUFFERS || B >= MAX_BUFFERS || C >= MAX_BUFFERS) {
        return SLOTH_VK_ERROR_INVALID_PARAM;
    }
    const float* a_ptr = (const float*)g_buffers[A].ptr;
    const float* b_ptr = (const float*)g_buffers[B].ptr;
    float* c_ptr = (float*)g_buffers[C].ptr;
    if (!a_ptr || !b_ptr || !c_ptr) return SLOTH_VK_ERROR_INVALID_PARAM;

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            double sum = 0.0;
            for (uint32_t k = 0; k < K; ++k) {
                sum += (double)a_ptr[m * K + k] * (double)b_ptr[k * N + n];
            }
            c_ptr[m * N + n] = (float)sum;
        }
    }
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_forward_lora(SlothBufferHandle X, SlothBufferHandle W_base, SlothBufferHandle A_lora, SlothBufferHandle B_lora, SlothBufferHandle Out, uint32_t batch, uint32_t seq, uint32_t in_dim, uint32_t out_dim, uint32_t rank, float alpha) {
    if (!X || !A_lora || !B_lora || !Out) return SLOTH_VK_ERROR_INVALID_PARAM;
    uint32_t T = batch * seq;
    const float* x_ptr = (const float*)g_buffers[X].ptr;
    const float* w_ptr = W_base ? (const float*)g_buffers[W_base].ptr : NULL;
    const float* a_ptr = (const float*)g_buffers[A_lora].ptr;
    const float* b_ptr = (const float*)g_buffers[B_lora].ptr;
    float* out_ptr = (float*)g_buffers[Out].ptr;
    if (!x_ptr || !a_ptr || !b_ptr || !out_ptr) return SLOTH_VK_ERROR_INVALID_PARAM;

    float scale = (rank > 0) ? (alpha / (float)rank) : 1.0f;
    float* h = (float*)malloc(T * rank * sizeof(float));
    if (!h) return SLOTH_VK_ERROR_OUT_OF_MEMORY;

    // h = X * A (X: T x in_dim, A: in_dim x rank)
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t r = 0; r < rank; ++r) {
            double sum = 0.0;
            for (uint32_t i = 0; i < in_dim; ++i) {
                sum += (double)x_ptr[t * in_dim + i] * (double)a_ptr[i * rank + r];
            }
            h[t * rank + r] = (float)sum;
        }
    }

    // Out = X * W_base + scale * h * B
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t o = 0; o < out_dim; ++o) {
            double base_val = 0.0;
            if (w_ptr) {
                for (uint32_t i = 0; i < in_dim; ++i) {
                    base_val += (double)x_ptr[t * in_dim + i] * (double)w_ptr[i * out_dim + o];
                }
            }
            double lora_val = 0.0;
            for (uint32_t r = 0; r < rank; ++r) {
                lora_val += (double)h[t * rank + r] * (double)b_ptr[r * out_dim + o];
            }
            out_ptr[t * out_dim + o] = (float)(base_val + scale * lora_val);
        }
    }

    free(h);
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_backward_lora(SlothBufferHandle X, SlothBufferHandle dOut, SlothBufferHandle A_lora, SlothBufferHandle B_lora, SlothBufferHandle dA_grad, SlothBufferHandle dB_grad, uint32_t batch, uint32_t seq, uint32_t in_dim, uint32_t out_dim, uint32_t rank, float alpha) {
    if (!X || !dOut || !A_lora || !B_lora || !dA_grad || !dB_grad) return SLOTH_VK_ERROR_INVALID_PARAM;
    uint32_t T = batch * seq;
    const float* x_ptr = (const float*)g_buffers[X].ptr;
    const float* dout_ptr = (const float*)g_buffers[dOut].ptr;
    const float* a_ptr = (const float*)g_buffers[A_lora].ptr;
    const float* b_ptr = (const float*)g_buffers[B_lora].ptr;
    float* da_ptr = (float*)g_buffers[dA_grad].ptr;
    float* db_ptr = (float*)g_buffers[dB_grad].ptr;
    if (!x_ptr || !dout_ptr || !a_ptr || !b_ptr || !da_ptr || !db_ptr) return SLOTH_VK_ERROR_INVALID_PARAM;

    float scale = (rank > 0) ? (alpha / (float)rank) : 1.0f;
    float* h = (float*)malloc(T * rank * sizeof(float));
    float* dh = (float*)malloc(T * rank * sizeof(float));
    if (!h || !dh) {
        if (h) free(h);
        if (dh) free(dh);
        return SLOTH_VK_ERROR_OUT_OF_MEMORY;
    }

    // 1. Forward intermediate: h = X * A (T x rank)
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t r = 0; r < rank; ++r) {
            double sum = 0.0;
            for (uint32_t i = 0; i < in_dim; ++i) {
                sum += (double)x_ptr[t * in_dim + i] * (double)a_ptr[i * rank + r];
            }
            h[t * rank + r] = (float)sum;
        }
    }

    // 2. dB_grad = scale * (h^T * dOut)
    for (uint32_t r = 0; r < rank; ++r) {
        for (uint32_t o = 0; o < out_dim; ++o) {
            double sum = 0.0;
            for (uint32_t t = 0; t < T; ++t) {
                sum += (double)h[t * rank + r] * (double)dout_ptr[t * out_dim + o];
            }
            db_ptr[r * out_dim + o] += (float)(scale * sum);
        }
    }

    // 3. dh = scale * (dOut * B^T)
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t r = 0; r < rank; ++r) {
            double sum = 0.0;
            for (uint32_t o = 0; o < out_dim; ++o) {
                sum += (double)dout_ptr[t * out_dim + o] * (double)b_ptr[r * out_dim + o];
            }
            dh[t * rank + r] = (float)(scale * sum);
        }
    }

    // 4. dA_grad = X^T * dh
    for (uint32_t i = 0; i < in_dim; ++i) {
        for (uint32_t r = 0; r < rank; ++r) {
            double sum = 0.0;
            for (uint32_t t = 0; t < T; ++t) {
                sum += (double)x_ptr[t * in_dim + i] * (double)dh[t * rank + r];
            }
            da_ptr[i * rank + r] += (float)sum;
        }
    }

    free(h);
    free(dh);
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_adamw_step(SlothBufferHandle Weights, SlothBufferHandle Grads, SlothBufferHandle M_state, SlothBufferHandle V_state, uint32_t num_elements, float lr, float beta1, float beta2, float eps, float weight_decay, uint32_t step) {
    if (!Weights || !Grads || !M_state || !V_state) return SLOTH_VK_ERROR_INVALID_PARAM;
    float* w = (float*)g_buffers[Weights].ptr;
    const float* g = (const float*)g_buffers[Grads].ptr;
    float* m = (float*)g_buffers[M_state].ptr;
    float* v = (float*)g_buffers[V_state].ptr;
    if (!w || !g || !m || !v) return SLOTH_VK_ERROR_INVALID_PARAM;

    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);
    if (bias_correction1 <= 0.0f) bias_correction1 = 1e-7f;
    if (bias_correction2 <= 0.0f) bias_correction2 = 1e-7f;

    for (uint32_t i = 0; i < num_elements; ++i) {
        // Weight decay
        w[i] -= lr * weight_decay * w[i];

        // Momentum updates
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * (g[i] * g[i]);

        // Bias-corrected estimates
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;

        // Parameter update
        w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_rmsnorm_forward(SlothBufferHandle X, SlothBufferHandle Gamma, SlothBufferHandle Out, uint32_t batch_seq, uint32_t dim, float eps) {
    if (!X || !Gamma || !Out) return SLOTH_VK_ERROR_INVALID_PARAM;
    const float* x = (const float*)g_buffers[X].ptr;
    const float* gamma = (const float*)g_buffers[Gamma].ptr;
    float* out = (float*)g_buffers[Out].ptr;
    if (!x || !gamma || !out) return SLOTH_VK_ERROR_INVALID_PARAM;

    for (uint32_t b = 0; b < batch_seq; ++b) {
        double sum_sq = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            float val = x[b * dim + d];
            sum_sq += (double)(val * val);
        }
        float rms = 1.0f / sqrtf((float)(sum_sq / dim) + eps);
        for (uint32_t d = 0; d < dim; ++d) {
            out[b * dim + d] = x[b * dim + d] * rms * gamma[d];
        }
    }
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_rmsnorm_backward(SlothBufferHandle dOut, SlothBufferHandle X, SlothBufferHandle Gamma, SlothBufferHandle dX, SlothBufferHandle dGamma, uint32_t batch_seq, uint32_t dim, float eps) {
    if (!dOut || !X || !Gamma || !dX || !dGamma) return SLOTH_VK_ERROR_INVALID_PARAM;
    const float* dout = (const float*)g_buffers[dOut].ptr;
    const float* x = (const float*)g_buffers[X].ptr;
    const float* gamma = (const float*)g_buffers[Gamma].ptr;
    float* dx = (float*)g_buffers[dX].ptr;
    float* dgamma = (float*)g_buffers[dGamma].ptr;
    if (!dout || !x || !gamma || !dx || !dgamma) return SLOTH_VK_ERROR_INVALID_PARAM;

    for (uint32_t b = 0; b < batch_seq; ++b) {
        double sum_sq = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            float val = x[b * dim + d];
            sum_sq += (double)(val * val);
        }
        float mean_sq = (float)(sum_sq / dim) + eps;
        float rms = 1.0f / sqrtf(mean_sq);
        float rms3 = rms * rms * rms;

        double sum_dout_gamma_x = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            sum_dout_gamma_x += (double)(dout[b * dim + d] * gamma[d] * x[b * dim + d]);
            dgamma[d] += dout[b * dim + d] * x[b * dim + d] * rms;
        }

        for (uint32_t d = 0; d < dim; ++d) {
            dx[b * dim + d] = (dout[b * dim + d] * gamma[d] * rms) - (float)(x[b * dim + d] * rms3 * sum_dout_gamma_x / dim);
        }
    }
    return SLOTH_VK_SUCCESS;
}
