/* CPU-заглушка Vulkan-слоя: собирается, когда загрузчик Vulkan не найден (или SLOTH_VULKAN=stub).
 *
 * Считает то же, что шейдеры, в той же раскладке матриц и с той же семантикой, поэтому код
 * поверх неё ведёт себя как на видеокарте. Раньше заглушка хранила LoRA транспонированной
 * (A [in, rank] вместо [rank, in]), накапливала градиенты вместо перезаписи и выдавала себя
 * за видеокарту на 4 ГиБ. Проверяется тем же тестом tests/gpu_vs_cpu.rs, что и GPU. */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/sloth_vulkan.h"

#define MAX_BUFFERS 4096

typedef struct {
    void* ptr;
    size_t size;
    int in_use;
} BufferEntry;

static BufferEntry g_buffers[MAX_BUFFERS] = {0};
static uint64_t g_allocated_bytes = 0;
static int g_initialized = 0;

static BufferEntry* find_entry(SlothBufferHandle handle) {
    if (handle == SLOTH_NULL_BUFFER || handle >= MAX_BUFFERS || !g_buffers[handle].in_use) {
        return NULL;
    }
    return &g_buffers[handle];
}

/* Указатель на массив float, если буфер существует и вмещает count элементов. */
static float* float_array(SlothBufferHandle handle, size_t count) {
    BufferEntry* entry = find_entry(handle);
    if (!entry || entry->size / sizeof(float) < count) return NULL;
    return (float*)entry->ptr;
}

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
        if (g_buffers[i].in_use) {
            free(g_buffers[i].ptr);
            g_buffers[i].ptr = NULL;
            g_buffers[i].size = 0;
            g_buffers[i].in_use = 0;
        }
    }
    g_allocated_bytes = 0;
    g_initialized = 0;
}

int sloth_vk_get_vram_info(uint64_t* total_bytes, uint64_t* used_bytes, uint64_t* free_bytes) {
    if (!g_initialized) return SLOTH_VK_ERROR_NOT_INITIALIZED;
    /* Видеопамяти нет: общий объём 0, а «занято» — сколько выделено в обычной памяти */
    if (total_bytes) *total_bytes = 0;
    if (used_bytes) *used_bytes = g_allocated_bytes;
    if (free_bytes) *free_bytes = 0;
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_get_device_info(SlothDeviceInfo* out) {
    if (!out) return SLOTH_VK_ERROR_INVALID_PARAM;
    if (!g_initialized) return SLOTH_VK_ERROR_NOT_INITIALIZED;
    memset(out, 0, sizeof(*out));
    strncpy(out->device_name, "SlothForge CPU fallback (no Vulkan)", SLOTH_VK_INFO_STRING_SIZE - 1);
    out->device_type = SLOTH_VK_DEVICE_TYPE_CPU;
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_get_memory_budget(uint64_t* budget_bytes, uint64_t* usage_bytes) {
    if (!budget_bytes || !usage_bytes) return SLOTH_VK_ERROR_INVALID_PARAM;
    return SLOTH_VK_ERROR_NOT_SUPPORTED;
}

SlothBufferHandle sloth_vk_alloc_buffer(size_t size_bytes, int is_device_local) {
    (void)is_device_local;
    if (!g_initialized) return SLOTH_NULL_BUFFER;
    for (size_t i = 1; i < MAX_BUFFERS; ++i) {
        if (!g_buffers[i].in_use) {
            /* calloc(0) может вернуть NULL, а пустой буфер — допустимый запрос */
            void* p = calloc(1, size_bytes > 0 ? size_bytes : 1);
            if (!p) return SLOTH_NULL_BUFFER;
            g_buffers[i].ptr = p;
            g_buffers[i].size = size_bytes;
            g_buffers[i].in_use = 1;
            g_allocated_bytes += size_bytes;
            return (SlothBufferHandle)i;
        }
    }
    return SLOTH_NULL_BUFFER;
}

void sloth_vk_free_buffer(SlothBufferHandle handle) {
    BufferEntry* entry = find_entry(handle);
    if (!entry) return;
    g_allocated_bytes = g_allocated_bytes >= entry->size ? g_allocated_bytes - entry->size : 0;
    free(entry->ptr);
    entry->ptr = NULL;
    entry->size = 0;
    entry->in_use = 0;
}

int sloth_vk_write_buffer(SlothBufferHandle handle, const void* src, size_t size_bytes) {
    if (!src || size_bytes == 0) return SLOTH_VK_SUCCESS;
    BufferEntry* entry = find_entry(handle);
    /* Как на GPU: запись больше буфера — ошибка, а не молчаливое обрезание */
    if (!entry || size_bytes > entry->size) return SLOTH_VK_ERROR_INVALID_PARAM;
    memcpy(entry->ptr, src, size_bytes);
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_read_buffer(SlothBufferHandle handle, void* dst, size_t size_bytes) {
    if (!dst || size_bytes == 0) return SLOTH_VK_SUCCESS;
    BufferEntry* entry = find_entry(handle);
    if (!entry || size_bytes > entry->size) return SLOTH_VK_ERROR_INVALID_PARAM;
    memcpy(dst, entry->ptr, size_bytes);
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_forward_gemm(SlothBufferHandle A, SlothBufferHandle B, SlothBufferHandle C, uint32_t M, uint32_t K, uint32_t N) {
    if (M == 0 || K == 0 || N == 0) return SLOTH_VK_SUCCESS;
    const float* a = float_array(A, (size_t)M * K);
    const float* b = float_array(B, (size_t)K * N);
    float* c = float_array(C, (size_t)M * N);
    if (!a || !b || !c) return SLOTH_VK_ERROR_INVALID_PARAM;

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                sum += (double)a[m * K + k] * (double)b[k * N + n];
            }
            c[m * N + n] = (float)sum;
        }
    }
    return SLOTH_VK_SUCCESS;
}

/* h[t, r] = (A x_t)[r] в раскладке PEFT: A [rank, in_dim]. */
static void lora_project_down(const float* x, const float* a, float* h, size_t tokens, size_t in_dim, size_t rank) {
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t r = 0; r < rank; ++r) {
            double sum = 0.0;
            for (size_t i = 0; i < in_dim; ++i) {
                sum += (double)x[t * in_dim + i] * (double)a[r * in_dim + i];
            }
            h[t * rank + r] = (float)sum;
        }
    }
}

int sloth_vk_forward_lora(SlothBufferHandle X, SlothBufferHandle W_base, SlothBufferHandle A_lora, SlothBufferHandle B_lora, SlothBufferHandle Out, uint32_t batch, uint32_t seq, uint32_t in_dim, uint32_t out_dim, uint32_t rank, float alpha) {
    size_t tokens = (size_t)batch * seq;
    if (tokens == 0 || in_dim == 0 || out_dim == 0) return SLOTH_VK_SUCCESS;
    if (rank > SLOTH_VK_LORA_MAX_RANK) return SLOTH_VK_ERROR_INVALID_PARAM;

    const float* x = float_array(X, tokens * in_dim);
    const float* w = float_array(W_base, (size_t)out_dim * in_dim);
    const float* a = float_array(A_lora, (size_t)rank * in_dim);
    const float* b = float_array(B_lora, (size_t)out_dim * rank);
    float* out = float_array(Out, tokens * out_dim);
    if (!x || !a || !b || !out) return SLOTH_VK_ERROR_INVALID_PARAM;
    /* Нулевой дескриптор W означает «без базового веса», неверный — ошибка */
    if (W_base != SLOTH_NULL_BUFFER && !w) return SLOTH_VK_ERROR_INVALID_PARAM;

    float scale = rank > 0 ? alpha / (float)rank : 0.0f;
    float* h = (float*)malloc((tokens * rank > 0 ? tokens * rank : 1) * sizeof(float));
    if (!h) return SLOTH_VK_ERROR_OUT_OF_MEMORY;
    lora_project_down(x, a, h, tokens, in_dim, rank);

    /* Out[t, o] = (W x_t)[o] + scale * (B h_t)[o]; W [out_dim, in_dim], B [out_dim, rank] */
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t o = 0; o < out_dim; ++o) {
            double base = 0.0;
            if (w) {
                for (size_t i = 0; i < in_dim; ++i) {
                    base += (double)x[t * in_dim + i] * (double)w[o * in_dim + i];
                }
            }
            double lora = 0.0;
            for (size_t r = 0; r < rank; ++r) {
                lora += (double)b[o * rank + r] * (double)h[t * rank + r];
            }
            out[t * out_dim + o] = (float)(base + scale * lora);
        }
    }
    free(h);
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_backward_lora(SlothBufferHandle X, SlothBufferHandle dOut, SlothBufferHandle A_lora, SlothBufferHandle B_lora, SlothBufferHandle dA_grad, SlothBufferHandle dB_grad, uint32_t batch, uint32_t seq, uint32_t in_dim, uint32_t out_dim, uint32_t rank, float alpha) {
    size_t tokens = (size_t)batch * seq;
    if (tokens == 0 || in_dim == 0 || out_dim == 0 || rank == 0) return SLOTH_VK_SUCCESS;

    const float* x = float_array(X, tokens * in_dim);
    const float* dout = float_array(dOut, tokens * out_dim);
    const float* a = float_array(A_lora, (size_t)rank * in_dim);
    const float* b = float_array(B_lora, (size_t)out_dim * rank);
    float* da = float_array(dA_grad, (size_t)rank * in_dim);
    float* db = float_array(dB_grad, (size_t)out_dim * rank);
    if (!x || !dout || !a || !b || !da || !db) return SLOTH_VK_ERROR_INVALID_PARAM;

    float scale = alpha / (float)rank;
    float* h = (float*)malloc(tokens * rank * sizeof(float));
    float* dh = (float*)malloc(tokens * rank * sizeof(float));
    if (!h || !dh) {
        free(h);
        free(dh);
        return SLOTH_VK_ERROR_OUT_OF_MEMORY;
    }
    lora_project_down(x, a, h, tokens, in_dim, rank);

    /* dh[t, r] = (B^T dOut_t)[r] */
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t r = 0; r < rank; ++r) {
            double sum = 0.0;
            for (size_t o = 0; o < out_dim; ++o) {
                sum += (double)dout[t * out_dim + o] * (double)b[o * rank + r];
            }
            dh[t * rank + r] = (float)sum;
        }
    }

    /* Как шейдер, градиенты перезаписываются: накопление между микробатчами — дело вызывающего */
    for (size_t o = 0; o < out_dim; ++o) {
        for (size_t r = 0; r < rank; ++r) {
            double sum = 0.0;
            for (size_t t = 0; t < tokens; ++t) {
                sum += (double)dout[t * out_dim + o] * (double)h[t * rank + r];
            }
            db[o * rank + r] = (float)(scale * sum);
        }
    }
    for (size_t r = 0; r < rank; ++r) {
        for (size_t i = 0; i < in_dim; ++i) {
            double sum = 0.0;
            for (size_t t = 0; t < tokens; ++t) {
                sum += (double)dh[t * rank + r] * (double)x[t * in_dim + i];
            }
            da[r * in_dim + i] = (float)(scale * sum);
        }
    }

    free(h);
    free(dh);
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_adamw_step(SlothBufferHandle Weights, SlothBufferHandle Grads, SlothBufferHandle M_state, SlothBufferHandle V_state, uint32_t num_elements, float lr, float beta1, float beta2, float eps, float weight_decay, uint32_t step) {
    if (num_elements == 0) return SLOTH_VK_SUCCESS;
    float* w = float_array(Weights, num_elements);
    const float* g = float_array(Grads, num_elements);
    float* m = float_array(M_state, num_elements);
    float* v = float_array(V_state, num_elements);
    if (!w || !g || !m || !v) return SLOTH_VK_ERROR_INVALID_PARAM;

    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);
    if (bias_correction1 <= 0.0f) bias_correction1 = 1e-7f;
    if (bias_correction2 <= 0.0f) bias_correction2 = 1e-7f;

    for (size_t i = 0; i < num_elements; ++i) {
        w[i] -= lr * weight_decay * w[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * (g[i] * g[i]);
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;
        w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
    return SLOTH_VK_SUCCESS;
}

/* 1 / sqrt(mean(x^2) + eps) для строки длины dim. */
static float inverse_rms(const float* row, size_t dim, float eps) {
    double sum_sq = 0.0;
    for (size_t d = 0; d < dim; ++d) {
        sum_sq += (double)row[d] * (double)row[d];
    }
    return 1.0f / sqrtf((float)(sum_sq / (double)dim) + eps);
}

int sloth_vk_rmsnorm_forward(SlothBufferHandle X, SlothBufferHandle Gamma, SlothBufferHandle Out, uint32_t batch_seq, uint32_t dim, float eps) {
    if (batch_seq == 0 || dim == 0) return SLOTH_VK_SUCCESS;
    const float* x = float_array(X, (size_t)batch_seq * dim);
    const float* gamma = float_array(Gamma, dim);
    float* out = float_array(Out, (size_t)batch_seq * dim);
    if (!x || !gamma || !out) return SLOTH_VK_ERROR_INVALID_PARAM;

    for (size_t t = 0; t < batch_seq; ++t) {
        const float* row = x + t * dim;
        float inv_rms = inverse_rms(row, dim, eps);
        for (size_t d = 0; d < dim; ++d) {
            out[t * dim + d] = row[d] * inv_rms * gamma[d];
        }
    }
    return SLOTH_VK_SUCCESS;
}

int sloth_vk_rmsnorm_backward(SlothBufferHandle dOut, SlothBufferHandle X, SlothBufferHandle Gamma, SlothBufferHandle dX, SlothBufferHandle dGamma, uint32_t batch_seq, uint32_t dim, float eps) {
    if (batch_seq == 0 || dim == 0) return SLOTH_VK_SUCCESS;
    size_t total = (size_t)batch_seq * dim;
    const float* dout = float_array(dOut, total);
    const float* x = float_array(X, total);
    const float* gamma = float_array(Gamma, dim);
    float* dx = float_array(dX, total);
    float* dgamma = float_array(dGamma, dim);
    if (!dout || !x || !gamma || !dx || !dgamma) return SLOTH_VK_ERROR_INVALID_PARAM;

    /* Как шейдер, dGamma перезаписывается */
    memset(dgamma, 0, (size_t)dim * sizeof(float));
    for (size_t t = 0; t < batch_seq; ++t) {
        const float* row = x + t * dim;
        const float* grad_row = dout + t * dim;
        float inv_rms = inverse_rms(row, dim, eps);
        float inv_rms3 = inv_rms * inv_rms * inv_rms;

        double dot = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            dot += (double)grad_row[d] * (double)gamma[d] * (double)row[d];
            dgamma[d] += grad_row[d] * row[d] * inv_rms;
        }
        for (size_t d = 0; d < dim; ++d) {
            dx[t * dim + d] = grad_row[d] * gamma[d] * inv_rms - (float)((double)row[d] * inv_rms3 * dot / (double)dim);
        }
    }
    return SLOTH_VK_SUCCESS;
}
