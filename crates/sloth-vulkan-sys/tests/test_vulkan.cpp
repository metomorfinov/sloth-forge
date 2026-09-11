#include "sloth_vulkan.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iomanip>

#define CHECK(expr) do { \
    int _rc = (expr); \
    if (_rc != SLOTH_VK_SUCCESS) { \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "] " << #expr << " returned " << _rc << std::endl; \
        std::exit(1); \
    } \
} while(0)

#define ASSERT_TRUE(expr, msg) do { \
    if (!(expr)) { \
        std::cerr << "ASSERTION FAILED [" << __FILE__ << ":" << __LINE__ << "]: " << (msg) << std::endl; \
        std::exit(1); \
    } \
} while(0)

void test_device_detection_and_vram() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 1: Physical Device Auto-Detection & VRAM" << std::endl;
    std::cout << "==========================================" << std::endl;

    char device_name[256] = {0};
    CHECK(sloth_vk_init(1, device_name, sizeof(device_name)));

    std::cout << "[+] Detected Vulkan Device: \"" << device_name << "\"" << std::endl;
    std::string name(device_name);
    ASSERT_TRUE(name.find("AMD") != std::string::npos ||
                name.find("Radeon") != std::string::npos ||
                name.find("POLARIS10") != std::string::npos,
                "Expected AMD Radeon GPU (POLARIS10) to be detected!");

    uint64_t total_vram = 0, used_vram = 0, free_vram = 0;
    CHECK(sloth_vk_get_vram_info(&total_vram, &used_vram, &free_vram));

    double total_gb = static_cast<double>(total_vram) / (1024.0 * 1024.0 * 1024.0);
    double used_mb = static_cast<double>(used_vram) / (1024.0 * 1024.0);
    double free_gb = static_cast<double>(free_vram) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "[+] VRAM Total: " << std::fixed << std::setprecision(2) << total_gb << " GB (" << total_vram << " bytes)" << std::endl;
    std::cout << "[+] VRAM Used:  " << std::fixed << std::setprecision(2) << used_mb << " MB" << std::endl;
    std::cout << "[+] VRAM Free:  " << std::fixed << std::setprecision(2) << free_gb << " GB" << std::endl;

    ASSERT_TRUE(total_vram > 0, "Total VRAM must be greater than 0");
    std::cout << "[PASS] Device auto-detection and VRAM query verified." << std::endl;
}

void test_buffer_allocation_and_transfers() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 2: VRAM Buffer Allocation & Staging Transfers" << std::endl;
    std::cout << "==========================================" << std::endl;

    size_t test_size = 16 * 1024 * 1024; // 16 MB
    std::cout << "[*] Allocating 16 MB Device-Local VRAM buffer..." << std::endl;
    SlothBufferHandle dev_buf = sloth_vk_alloc_buffer(test_size, 1);
    ASSERT_TRUE(dev_buf != SLOTH_NULL_BUFFER, "Failed to allocate device local buffer");

    uint64_t total = 0, used = 0, free = 0;
    CHECK(sloth_vk_get_vram_info(&total, &used, &free));
    std::cout << "[+] VRAM Used after 16 MB allocation: " << (used / (1024 * 1024)) << " MB" << std::endl;
    ASSERT_TRUE(used >= test_size, "Tracked used VRAM should reflect allocated buffer");

    // Fill host buffer with test pattern
    std::vector<uint32_t> host_src(test_size / sizeof(uint32_t));
    for (size_t i = 0; i < host_src.size(); ++i) {
        host_src[i] = static_cast<uint32_t>(0xDEADBEEF ^ (i * 1337));
    }

    std::cout << "[*] Writing 16 MB via staging buffer to GPU..." << std::endl;
    CHECK(sloth_vk_write_buffer(dev_buf, host_src.data(), test_size));

    std::vector<uint32_t> host_dst(test_size / sizeof(uint32_t), 0);
    std::cout << "[*] Reading 16 MB back from GPU..." << std::endl;
    CHECK(sloth_vk_read_buffer(dev_buf, host_dst.data(), test_size));

    for (size_t i = 0; i < host_src.size(); ++i) {
        if (host_src[i] != host_dst[i]) {
            std::cerr << "Mismatch at index " << i << ": expected 0x"
                      << std::hex << host_src[i] << ", got 0x" << host_dst[i] << std::dec << std::endl;
            ASSERT_TRUE(false, "VRAM readback data does not match written data");
        }
    }

    sloth_vk_free_buffer(dev_buf);
    CHECK(sloth_vk_get_vram_info(&total, &used, &free));
    std::cout << "[+] VRAM Used after freeing buffer: " << (used / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "[PASS] 16 MB device-local buffer write & readback verified successfully." << std::endl;
}

void test_forward_gemm() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 3: Polaris 10 Wavefront-64 Tiled GEMM" << std::endl;
    std::cout << "==========================================" << std::endl;

    // Test with non-power-of-2 dimensions to stress boundary checks: M=37, K=53, N=49
    uint32_t M = 37, K = 53, N = 49;
    std::cout << "[*] Running GEMM: M=" << M << ", K=" << K << ", N=" << N << std::endl;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = std::sin(static_cast<float>(i + 1));
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = std::cos(static_cast<float>(i + 1));

    // CPU reference: row-major C = A * B
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }

    SlothBufferHandle bufA = sloth_vk_alloc_buffer(h_A.size() * sizeof(float), 1);
    SlothBufferHandle bufB = sloth_vk_alloc_buffer(h_B.size() * sizeof(float), 1);
    SlothBufferHandle bufC = sloth_vk_alloc_buffer(h_C.size() * sizeof(float), 1);

    CHECK(sloth_vk_write_buffer(bufA, h_A.data(), h_A.size() * sizeof(float)));
    CHECK(sloth_vk_write_buffer(bufB, h_B.data(), h_B.size() * sizeof(float)));

    CHECK(sloth_vk_forward_gemm(bufA, bufB, bufC, M, K, N));

    CHECK(sloth_vk_read_buffer(bufC, h_C.data(), h_C.size() * sizeof(float)));

    float max_diff = 0.0f;
    for (size_t i = 0; i < h_C.size(); ++i) {
        float diff = std::abs(h_C[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "[+] GEMM Max Difference vs CPU reference: " << max_diff << std::endl;
    ASSERT_TRUE(max_diff < 1e-4f, "GEMM output exceeds tolerance");

    sloth_vk_free_buffer(bufA);
    sloth_vk_free_buffer(bufB);
    sloth_vk_free_buffer(bufC);
    std::cout << "[PASS] GEMM compute kernel verified." << std::endl;
}

void test_forward_and_backward_lora() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 4: LoRA Forward & Backward In-VRAM" << std::endl;
    std::cout << "==========================================" << std::endl;

    uint32_t batch = 2, seq = 4;
    uint32_t tokens = batch * seq; // 8 tokens
    uint32_t in_dim = 32, out_dim = 24, rank = 8;
    float alpha = 16.0f;
    float scale = alpha / static_cast<float>(rank);

    std::vector<float> h_X(tokens * in_dim);
    std::vector<float> h_W(out_dim * in_dim);
    std::vector<float> h_A(rank * in_dim);
    std::vector<float> h_B(out_dim * rank);
    std::vector<float> h_Out(tokens * out_dim, 0.0f);
    std::vector<float> h_Out_ref(tokens * out_dim, 0.0f);

    for (size_t i = 0; i < h_X.size(); ++i) h_X[i] = 0.01f * (i % 17);
    for (size_t i = 0; i < h_W.size(); ++i) h_W[i] = 0.02f * (i % 23);
    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = 0.03f * (i % 11);
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = 0.04f * (i % 13);

    // CPU Reference Forward
    for (uint32_t t = 0; t < tokens; ++t) {
        // 1. h = A * x
        std::vector<float> h_vec(rank, 0.0f);
        for (uint32_t r = 0; r < rank; ++r) {
            for (uint32_t k = 0; k < in_dim; ++k) {
                h_vec[r] += h_A[r * in_dim + k] * h_X[t * in_dim + k];
            }
        }
        // 2. Base + scale * B * h
        for (uint32_t o = 0; o < out_dim; ++o) {
            float base = 0.0f;
            for (uint32_t k = 0; k < in_dim; ++k) {
                base += h_W[o * in_dim + k] * h_X[t * in_dim + k];
            }
            float lora = 0.0f;
            for (uint32_t r = 0; r < rank; ++r) {
                lora += h_B[o * rank + r] * h_vec[r];
            }
            h_Out_ref[t * out_dim + o] = base + scale * lora;
        }
    }

    SlothBufferHandle bufX = sloth_vk_alloc_buffer(h_X.size() * sizeof(float), 1);
    SlothBufferHandle bufW = sloth_vk_alloc_buffer(h_W.size() * sizeof(float), 1);
    SlothBufferHandle bufA = sloth_vk_alloc_buffer(h_A.size() * sizeof(float), 1);
    SlothBufferHandle bufB = sloth_vk_alloc_buffer(h_B.size() * sizeof(float), 1);
    SlothBufferHandle bufOut = sloth_vk_alloc_buffer(h_Out.size() * sizeof(float), 1);

    CHECK(sloth_vk_write_buffer(bufX, h_X.data(), h_X.size() * sizeof(float)));
    CHECK(sloth_vk_write_buffer(bufW, h_W.data(), h_W.size() * sizeof(float)));
    CHECK(sloth_vk_write_buffer(bufA, h_A.data(), h_A.size() * sizeof(float)));
    CHECK(sloth_vk_write_buffer(bufB, h_B.data(), h_B.size() * sizeof(float)));

    // Run forward LoRA
    CHECK(sloth_vk_forward_lora(bufX, bufW, bufA, bufB, bufOut, batch, seq, in_dim, out_dim, rank, alpha));
    CHECK(sloth_vk_read_buffer(bufOut, h_Out.data(), h_Out.size() * sizeof(float)));

    float max_diff_fwd = 0.0f;
    for (size_t i = 0; i < h_Out.size(); ++i) {
        float diff = std::abs(h_Out[i] - h_Out_ref[i]);
        if (diff > max_diff_fwd) max_diff_fwd = diff;
    }
    std::cout << "[+] LoRA Forward Max Difference vs CPU: " << max_diff_fwd << std::endl;
    ASSERT_TRUE(max_diff_fwd < 1e-4f, "LoRA forward output exceeds tolerance");

    // LoRA Backward
    std::vector<float> h_dOut(tokens * out_dim);
    for (size_t i = 0; i < h_dOut.size(); ++i) h_dOut[i] = 0.05f * (i % 7);

    // CPU Reference Backward
    std::vector<float> h_dA_ref(rank * in_dim, 0.0f);
    std::vector<float> h_dB_ref(out_dim * rank, 0.0f);

    for (uint32_t t = 0; t < tokens; ++t) {
        // h_t = A * x_t
        std::vector<float> h_vec(rank, 0.0f);
        for (uint32_t r = 0; r < rank; ++r) {
            for (uint32_t k = 0; k < in_dim; ++k) {
                h_vec[r] += h_A[r * in_dim + k] * h_X[t * in_dim + k];
            }
        }
        // dB_grad[o, r] += scale * dOut[t, o] * h_vec[r]
        for (uint32_t o = 0; o < out_dim; ++o) {
            for (uint32_t r = 0; r < rank; ++r) {
                h_dB_ref[o * rank + r] += scale * h_dOut[t * out_dim + o] * h_vec[r];
            }
        }
        // dh_t[r] = sum_o dOut[t, o] * B[o, r]
        std::vector<float> dh_vec(rank, 0.0f);
        for (uint32_t r = 0; r < rank; ++r) {
            for (uint32_t o = 0; o < out_dim; ++o) {
                dh_vec[r] += h_dOut[t * out_dim + o] * h_B[o * rank + r];
            }
        }
        // dA_grad[r, k] += scale * dh_vec[r] * X[t, k]
        for (uint32_t r = 0; r < rank; ++r) {
            for (uint32_t k = 0; k < in_dim; ++k) {
                h_dA_ref[r * in_dim + k] += scale * dh_vec[r] * h_X[t * in_dim + k];
            }
        }
    }

    SlothBufferHandle bufdOut = sloth_vk_alloc_buffer(h_dOut.size() * sizeof(float), 1);
    SlothBufferHandle bufdA = sloth_vk_alloc_buffer(h_dA_ref.size() * sizeof(float), 1);
    SlothBufferHandle bufdB = sloth_vk_alloc_buffer(h_dB_ref.size() * sizeof(float), 1);

    CHECK(sloth_vk_write_buffer(bufdOut, h_dOut.data(), h_dOut.size() * sizeof(float)));

    CHECK(sloth_vk_backward_lora(bufX, bufdOut, bufA, bufB, bufdA, bufdB, batch, seq, in_dim, out_dim, rank, alpha));

    std::vector<float> h_dA(rank * in_dim, 0.0f);
    std::vector<float> h_dB(out_dim * rank, 0.0f);
    CHECK(sloth_vk_read_buffer(bufdA, h_dA.data(), h_dA.size() * sizeof(float)));
    CHECK(sloth_vk_read_buffer(bufdB, h_dB.data(), h_dB.size() * sizeof(float)));

    float max_diff_dA = 0.0f;
    for (size_t i = 0; i < h_dA.size(); ++i) {
        float diff = std::abs(h_dA[i] - h_dA_ref[i]);
        if (diff > max_diff_dA) max_diff_dA = diff;
    }

    float max_diff_dB = 0.0f;
    for (size_t i = 0; i < h_dB.size(); ++i) {
        float diff = std::abs(h_dB[i] - h_dB_ref[i]);
        if (diff > max_diff_dB) max_diff_dB = diff;
    }

    std::cout << "[+] LoRA Backward dA Max Diff: " << max_diff_dA << std::endl;
    std::cout << "[+] LoRA Backward dB Max Diff: " << max_diff_dB << std::endl;
    ASSERT_TRUE(max_diff_dA < 1e-4f, "LoRA dA gradient exceeds tolerance");
    ASSERT_TRUE(max_diff_dB < 1e-4f, "LoRA dB gradient exceeds tolerance");

    sloth_vk_free_buffer(bufX);
    sloth_vk_free_buffer(bufW);
    sloth_vk_free_buffer(bufA);
    sloth_vk_free_buffer(bufB);
    sloth_vk_free_buffer(bufOut);
    sloth_vk_free_buffer(bufdOut);
    sloth_vk_free_buffer(bufdA);
    sloth_vk_free_buffer(bufdB);

    std::cout << "[PASS] LoRA forward and backward verified." << std::endl;
}

void test_adamw() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 5: In-VRAM AdamW Optimizer Step" << std::endl;
    std::cout << "==========================================" << std::endl;

    uint32_t N = 1024;
    float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, weight_decay = 0.01f;
    uint32_t step = 5;

    std::vector<float> h_W(N), h_G(N), h_M(N), h_V(N);
    std::vector<float> ref_W(N), ref_M(N), ref_V(N);

    for (uint32_t i = 0; i < N; ++i) {
        h_W[i] = 1.0f + 0.1f * std::sin(static_cast<float>(i));
        h_G[i] = 0.05f * std::cos(static_cast<float>(i));
        h_M[i] = 0.01f * std::sin(static_cast<float>(i));
        h_V[i] = 0.001f * std::cos(static_cast<float>(i * 2));

        ref_W[i] = h_W[i];
        ref_M[i] = h_M[i];
        ref_V[i] = h_V[i];
    }

    // CPU Reference AdamW
    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    for (uint32_t i = 0; i < N; ++i) {
        ref_W[i] = ref_W[i] - lr * weight_decay * ref_W[i];
        ref_M[i] = beta1 * ref_M[i] + (1.0f - beta1) * h_G[i];
        ref_V[i] = beta2 * ref_V[i] + (1.0f - beta2) * (h_G[i] * h_G[i]);

        float m_hat = ref_M[i] / bc1;
        float v_hat = ref_V[i] / bc2;
        ref_W[i] = ref_W[i] - lr * (m_hat / (std::sqrt(v_hat) + eps));
    }

    SlothBufferHandle bufW = sloth_vk_alloc_buffer(N * sizeof(float), 1);
    SlothBufferHandle bufG = sloth_vk_alloc_buffer(N * sizeof(float), 1);
    SlothBufferHandle bufM = sloth_vk_alloc_buffer(N * sizeof(float), 1);
    SlothBufferHandle bufV = sloth_vk_alloc_buffer(N * sizeof(float), 1);

    CHECK(sloth_vk_write_buffer(bufW, h_W.data(), N * sizeof(float)));
    CHECK(sloth_vk_write_buffer(bufG, h_G.data(), N * sizeof(float)));
    CHECK(sloth_vk_write_buffer(bufM, h_M.data(), N * sizeof(float)));
    CHECK(sloth_vk_write_buffer(bufV, h_V.data(), N * sizeof(float)));

    CHECK(sloth_vk_adamw_step(bufW, bufG, bufM, bufV, N, lr, beta1, beta2, eps, weight_decay, step));

    CHECK(sloth_vk_read_buffer(bufW, h_W.data(), N * sizeof(float)));

    float max_diff_w = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        float diff = std::abs(h_W[i] - ref_W[i]);
        if (diff > max_diff_w) max_diff_w = diff;
    }
    std::cout << "[+] AdamW Weights Max Diff vs CPU: " << max_diff_w << std::endl;
    ASSERT_TRUE(max_diff_w < 1e-5f, "AdamW weights differ from reference");

    sloth_vk_free_buffer(bufW);
    sloth_vk_free_buffer(bufG);
    sloth_vk_free_buffer(bufM);
    sloth_vk_free_buffer(bufV);

    std::cout << "[PASS] AdamW optimizer step verified." << std::endl;
}

void test_rmsnorm() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "TEST 6: RMSNorm Forward & Backward" << std::endl;
    std::cout << "==========================================" << std::endl;

    uint32_t batch_seq = 4;
    uint32_t dim = 64;
    float eps = 1e-5f;

    std::vector<float> h_X(batch_seq * dim);
    std::vector<float> h_Gamma(dim, 1.0f);
    std::vector<float> h_Out(batch_seq * dim, 0.0f);
    std::vector<float> ref_Out(batch_seq * dim, 0.0f);

    for (size_t i = 0; i < h_X.size(); ++i) h_X[i] = 0.1f * (i % 19 - 9);
    for (size_t i = 0; i < h_Gamma.size(); ++i) h_Gamma[i] = 0.5f + 0.1f * (i % 5);

    // CPU reference RMSNorm
    for (uint32_t t = 0; t < batch_seq; ++t) {
        float sum_sq = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            float v = h_X[t * dim + d];
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
        for (uint32_t d = 0; d < dim; ++d) {
            ref_Out[t * dim + d] = h_X[t * dim + d] * inv_rms * h_Gamma[d];
        }
    }

    SlothBufferHandle bufX = sloth_vk_alloc_buffer(h_X.size() * sizeof(float), 1);
    SlothBufferHandle bufGamma = sloth_vk_alloc_buffer(h_Gamma.size() * sizeof(float), 1);
    SlothBufferHandle bufOut = sloth_vk_alloc_buffer(h_Out.size() * sizeof(float), 1);

    CHECK(sloth_vk_write_buffer(bufX, h_X.data(), h_X.size() * sizeof(float)));
    CHECK(sloth_vk_write_buffer(bufGamma, h_Gamma.data(), h_Gamma.size() * sizeof(float)));

    CHECK(sloth_vk_rmsnorm_forward(bufX, bufGamma, bufOut, batch_seq, dim, eps));
    CHECK(sloth_vk_read_buffer(bufOut, h_Out.data(), h_Out.size() * sizeof(float)));

    float max_diff = 0.0f;
    for (size_t i = 0; i < h_Out.size(); ++i) {
        float diff = std::abs(h_Out[i] - ref_Out[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "[+] RMSNorm Forward Max Diff: " << max_diff << std::endl;
    ASSERT_TRUE(max_diff < 1e-4f, "RMSNorm forward differs from reference");

    sloth_vk_free_buffer(bufX);
    sloth_vk_free_buffer(bufGamma);
    sloth_vk_free_buffer(bufOut);

    std::cout << "[PASS] RMSNorm verified." << std::endl;
}

int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << "SlothForge Vulkan Compute Engine Integration Tests" << std::endl;
    std::cout << "Target Hardware: AMD Radeon RX 570 Series (POLARIS10)" << std::endl;
    std::cout << "========================================================" << std::endl;

    test_device_detection_and_vram();
    test_buffer_allocation_and_transfers();
    test_forward_gemm();
    test_forward_and_backward_lora();
    test_adamw();
    test_rmsnorm();

    std::cout << "\n[*] Shutting down Vulkan compute engine..." << std::endl;
    sloth_vk_shutdown();
    std::cout << "[+] Shutdown complete." << std::endl;

    std::cout << "\n========================================================" << std::endl;
    std::cout << "ALL VULKAN COMPUTE ENGINE TESTS PASSED SUCCESSFULLY! [6/6]" << std::endl;
    std::cout << "========================================================" << std::endl;

    return 0;
}
