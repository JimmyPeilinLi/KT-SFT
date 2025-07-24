//─────────────────────────────────────────────────────────────────────────────
//  File: test_sft_moe.hpp
//  Purpose : Minimal reproducible test for
//            BufferA/B/C  (from_mat → mat_mul → to_mat)
//  Author  : ChatGPT
//─────────────────────────────────────────────────────────────────────────────
#ifndef TEST_SFT_MOE_HPP
#define TEST_SFT_MOE_HPP

#include <immintrin.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <cmath>
#include <cassert>

#include "ggml.h"            // 提供 ggml_bf16_t 定义
#include "la/amx.hpp"        // amx::mat_mul 实现

//—— FP32 ↔ BF16 简易转换 ----------------------------------------------------
static inline ggml_bf16_t fp32_to_bf16(float x) {
    uint32_t tmp;
    std::memcpy(&tmp, &x, sizeof(tmp));
    return static_cast<ggml_bf16_t>(tmp >> 16);
}
static inline float bf16_to_fp32(ggml_bf16_t x) {
    uint32_t tmp = (static_cast<uint32_t>(x) << 16);
    float out;
    std::memcpy(&out, &tmp, sizeof(out));
    return out;
}
//─────────────────────────────────────────────────────────────────────────────
int main() {
    using Kernel = GemmKernel224Int8;

    constexpr int M = Kernel::M_STEP;   // 32
    constexpr int K = Kernel::K_STEP;   // 64
    constexpr int N = Kernel::N_STEP;   // 32

    //—— 1. 构造随机输入 ------------------------------------------------------
    std::vector<float> A_fp32(M * K);
    std::vector<float> B_fp32(N * K);   // 注意：B 是 (n, k) 排布
    std::vector<float> C_ref(M * N, 0.f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &v : A_fp32) v = dist(rng);
    for (auto &v : B_fp32) v = dist(rng);

    std::vector<ggml_bf16_t> A_bf16(M * K);
    std::vector<ggml_bf16_t> B_bf16(N * K);
    for (size_t i = 0; i < A_fp32.size(); ++i) A_bf16[i] = fp32_to_bf16(A_fp32[i]);
    for (size_t i = 0; i < B_fp32.size(); ++i) B_bf16[i] = fp32_to_bf16(B_fp32[i]);

    //—— 2. 计算 FP32 基准 ----------------------------------------------------
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                C_ref[m * N + n] += A_fp32[m * K + k] * B_fp32[n * K + k];

    //—— 3. 创建 Buffer 并执行 from_mat --------------------------------------
    void* bufA_raw = std::aligned_alloc(64, Kernel::BufferA::required_size(M, K));
    void* bufB_raw = std::aligned_alloc(64, Kernel::BufferB::required_size(N, K));
    void* bufC_raw = std::aligned_alloc(64, Kernel::BufferC::required_size(M, N));
    assert(bufA_raw && bufB_raw && bufC_raw);

    auto bufA = std::make_shared<Kernel::BufferA>(M, K, bufA_raw);
    auto bufB = std::make_shared<Kernel::BufferB>(N, K, bufB_raw);
    auto bufC = std::make_shared<Kernel::BufferC>(M, N, bufC_raw);

    bufA->from_mat(M, A_bf16.data(), 0, 1);   // 单线程 (ith=0, nth=1)
    bufB->from_mat(B_bf16.data(), 0, 1);

    //—— 4. 执行 AMX 矩阵乘 ---------------------------------------------------
    Kernel::config();               // 设置 tile-config（只需一次）
    const int nth = 1;              // 单线程示例
    const bool use_amx = true;      // 若硬件不支持，可改为 false
    amx::mat_mul(M, K, N, bufA, bufB, bufC,
                 /*ith=*/0, /*nth=*/nth, use_amx);

    //—— 5. to_mat 还原 BF16 结果 --------------------------------------------
    std::vector<ggml_bf16_t> C_bf16(M * N);
    bufC->to_mat(M, C_bf16.data(), 0, 1);

    //—— 6. 误差评估 ---------------------------------------------------------
    double mae = 0.0;
    for (int i = 0; i < M * N; ++i) {
        float v = bf16_to_fp32(C_bf16[i]);
        mae += std::abs(v - C_ref[i]);
    }
    mae /= (M * N);

    std::cout << "[test_sft_moe] mean abs error: " << mae << '\n';
    std::cout << ((mae < 1e-2) ? "PASS ✅" : "FAIL ❌") << std::endl;

    return (mae < 1e-2) ? 0 : 1;
}
//─────────────────────────────────────────────────────────────────────────────
#endif  // TEST_SFT_MOE_HPP
