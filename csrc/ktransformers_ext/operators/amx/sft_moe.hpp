/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2025-04-25 18:28:12
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2025-04-25 18:28:12
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_SFT_AMX_MOE_H
#define CPUINFER_OPERATOR_SFT_AMX_MOE_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>

#include "../../cpu_backend/backend.h"
#include "../../cpu_backend/shared_mem_buffer.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

#include "la/amx.hpp"

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
void *numa_alloc_aligned(size_t size, int node, size_t alignment) {
  void *ptr = numa_alloc_onnode(size, node);
  assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
  return ptr;
}
#endif

// static inline __m512 sft_exp_avx512(__m512 x) {
//   const __m512 log2e = _mm512_set1_ps(1.44269504089f);
//   const __m512 c1 = _mm512_set1_ps(0.69314718056f);

//   __m512 y = _mm512_mul_ps(x, log2e);
//   __m512i int_part = _mm512_cvtps_epi32(y);
//   __m512 frac_part = _mm512_sub_ps(y, _mm512_cvtepi32_ps(int_part));

//   const __m512 poly_1 = _mm512_set1_ps(0.9999999995f);
//   const __m512 poly_2 = _mm512_set1_ps(0.6931471805f);
//   const __m512 poly_3 = _mm512_set1_ps(0.2402265069f);
//   const __m512 poly_4 = _mm512_set1_ps(0.0555041087f);
//   const __m512 poly_5 = _mm512_set1_ps(0.0096181291f);
//   const __m512 poly_6 = _mm512_set1_ps(0.0013333558f);

//   __m512 frac_exp = _mm512_fmadd_ps(
//       frac_part, poly_6,
//       _mm512_fmadd_ps(frac_part, poly_5,
//                       _mm512_fmadd_ps(frac_part, poly_4,
//                                       _mm512_fmadd_ps(frac_part, poly_3, _mm512_fmadd_ps(frac_part, poly_2, poly_1)))));

//   __m512 two_pow_i = _mm512_scalef_ps(_mm512_set1_ps(1.0f), _mm512_cvtepi32_ps(int_part));
//   return _mm512_mul_ps(two_pow_i, frac_exp);
// }

// static inline __m512 sft_act_fn(__m512 gate_val, __m512 up_val) {
//   __m512 neg_gate_val = _mm512_sub_ps(_mm512_setzero_ps(), gate_val);
//   __m512 exp_neg_gate = sft_exp_avx512(neg_gate_val);
//   __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_gate);
//   __m512 act_val = _mm512_div_ps(gate_val, denom);

//   return _mm512_mul_ps(act_val, up_val);
// }
static inline __m512 sft_exp_avx512(__m512 x) {
  const __m512 log2e = _mm512_set1_ps(1.44269504089f);
  __m512 y = _mm512_mul_ps(x, log2e);
  __m512i i = _mm512_cvtps_epi32(y);
  __m512 f = _mm512_sub_ps(y, _mm512_cvtepi32_ps(i));
  const __m512 c0 = _mm512_set1_ps(0.9999999995f);
  const __m512 c1 = _mm512_set1_ps(0.6931471805f);
  const __m512 c2 = _mm512_set1_ps(0.2402265069f);
  const __m512 c3 = _mm512_set1_ps(0.0555041087f);
  const __m512 c4 = _mm512_set1_ps(0.0096181291f);
  const __m512 c5 = _mm512_set1_ps(0.0013333558f);
  __m512 poly = _mm512_fmadd_ps(f, c5,
                 _mm512_fmadd_ps(f, c4,
                 _mm512_fmadd_ps(f, c3,
                 _mm512_fmadd_ps(f, c2,
                 _mm512_fmadd_ps(f, c1, c0)))));
  __m512 two_i = _mm512_scalef_ps(_mm512_set1_ps(1.0f), _mm512_cvtepi32_ps(i));
  return _mm512_mul_ps(two_i, poly);
}
static inline __m512 sft_sigmoid(__m512 x) {
  __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), x);
  __m512 e = sft_exp_avx512(neg);
  __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), e);
  return _mm512_div_ps(_mm512_set1_ps(1.0f), denom);
}
static inline __m512 sft_act_fn(__m512 gate_val, __m512 up_val) {
  __m512 sig = sft_sigmoid(gate_val);
  __m512 swish = _mm512_mul_ps(gate_val, sig);        // Swish
  return _mm512_mul_ps(swish, up_val);                // Swish-GLU
}
static inline __m512 sft_act_fn_grad(__m512 gate_val) {
  __m512 sig = sft_sigmoid(gate_val);
  return _mm512_fmadd_ps(gate_val,
         _mm512_mul_ps(sig, _mm512_sub_ps(_mm512_set1_ps(1.0f), sig)),
         sig);                                         // derivative
}

struct SFT_AMX_MOEConfig {
  int expert_num;
  int routed_expert_num;
  int hidden_size;
  int intermediate_size;
  int max_len;
  void *gate_proj;
  void *up_proj;
  void *down_proj;

  SFT_AMX_MOEConfig() {}

  SFT_AMX_MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int max_len,
                void *gate_proj, void *up_proj, void *down_proj)
      : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size),
        intermediate_size(intermediate_size), max_len(max_len), gate_proj(gate_proj), up_proj(up_proj),
        down_proj(down_proj) {}
};

template <class T> class SFT_AMX_MOE {
private:
  SFT_AMX_MOEConfig config_;
  void *gate_proj_; // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
  void *up_proj_;   // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
  void *down_proj_; // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

  ggml_bf16_t *m_local_input_;       // [routed_expert_num * max_len * hidden_size]
  ggml_bf16_t *m_local_gate_output_; // [routed_expert_num * max_len * intermediate_size]
  ggml_bf16_t *m_local_up_output_;   // [routed_expert_num * max_len * intermediate_size]
  ggml_bf16_t *m_local_down_output_; // [routed_expert_num * max_len * hidden_size]

  std::vector<std::vector<int>> m_local_pos_;          // [max_len, routed_expert_num]
  std::vector<int> m_local_num_;                       // [expert_num]
  std::vector<int> m_expert_id_map_;                   // [expert_num]
  std::vector<ggml_bf16_t *> m_local_input_ptr_;       // [expert_num]
  std::vector<ggml_bf16_t *> m_local_gate_output_ptr_; // [expert_num]
  std::vector<ggml_bf16_t *> m_local_up_output_ptr_;   // [expert_num]
  std::vector<ggml_bf16_t *> m_local_down_output_ptr_; // [expert_num]

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

#ifdef USE_NUMA
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> gate_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> up_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> down_bb_numa_;
#else
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
#endif

  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_t_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_t_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_t_;

  ggml_bf16_t *m_local_down_grad_;          // dL/dy bucket
  float *m_local_v_grad_fp32_;              // after Down^T
  float *m_local_gate_grad_fp32_;           // dL/dg
  float *m_local_up_grad_fp32_;             // dL/du
  ggml_bf16_t *m_local_gate_grad_bf16_;
  ggml_bf16_t *m_local_up_grad_bf16_;
  float *m_local_gate_in_grad_fp32_;
  float *m_local_up_in_grad_fp32_;

  std::vector<ggml_bf16_t *> m_local_down_grad_ptr_;
  std::vector<float *> m_local_v_grad_ptr_;
  std::vector<float *> m_local_gate_grad_ptr_fp32_;
  std::vector<float *> m_local_up_grad_ptr_fp32_;
  std::vector<ggml_bf16_t *> m_local_gate_grad_ptr_bf16_;
  std::vector<ggml_bf16_t *> m_local_up_grad_ptr_bf16_;
  std::vector<float *> m_local_gate_in_grad_ptr_;
  std::vector<float *> m_local_up_in_grad_ptr_;

  struct FWDCacheRow {
    std::vector<float> gate_u;  // [intermediate_size]
    std::vector<float> up_v;    // [intermediate_size]
    FWDCacheRow() {}
    void init(int sz) {
      gate_u.resize(sz);
      up_v.resize(sz);
    }
  };
  std::vector<FWDCacheRow> m_forward_cache_;

public:
  SFT_AMX_MOE(SFT_AMX_MOEConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    std::vector<std::pair<void **, uint64_t>> m_mem_requests;
    m_mem_requests.push_back({(void **)&m_local_input_,
                              sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests.push_back({(void **)&m_local_gate_output_, sizeof(ggml_bf16_t) * config_.routed_expert_num *
                                                                  config_.max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void **)&m_local_up_output_, sizeof(ggml_bf16_t) * config_.routed_expert_num *
                                                                config_.max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void **)&m_local_down_output_,
                              sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    std::vector<void *> gate_up_ba_ptr(config_.expert_num);
    std::vector<void *> gate_bc_ptr(config_.expert_num);
    std::vector<void *> up_bc_ptr(config_.expert_num);
    std::vector<void *> down_ba_ptr(config_.expert_num);
    std::vector<void *> down_bc_ptr(config_.expert_num);

	// 反向额外的内存申请
	size_t bucket_cap = size_t(config_.routed_expert_num) * config_.max_len; // 最大 token×路由
    size_t bf16_bytes = sizeof(ggml_bf16_t);
    size_t f32_bytes  = sizeof(float);
    m_mem_requests.push_back({(void **)&m_local_down_grad_,
                              bf16_bytes * bucket_cap * config_.hidden_size});
    m_mem_requests.push_back(
        {(void **)&m_local_v_grad_fp32_,
         f32_bytes * bucket_cap * config_.intermediate_size});
    m_mem_requests.push_back(
        {(void **)&m_local_gate_grad_fp32_,
         f32_bytes * bucket_cap * config_.intermediate_size});
    m_mem_requests.push_back(
        {(void **)&m_local_up_grad_fp32_,
         f32_bytes * bucket_cap * config_.intermediate_size});
    m_mem_requests.push_back(
        {(void **)&m_local_gate_grad_bf16_,
         bf16_bytes * bucket_cap * config_.intermediate_size});
    m_mem_requests.push_back(
        {(void **)&m_local_up_grad_bf16_,
         bf16_bytes * bucket_cap * config_.intermediate_size});
    m_mem_requests.push_back(
        {(void **)&m_local_gate_in_grad_fp32_,
         f32_bytes * bucket_cap * config_.hidden_size});
    m_mem_requests.push_back(
        {(void **)&m_local_up_in_grad_fp32_,
         f32_bytes * bucket_cap * config_.hidden_size});
    for (int i = 0; i < config_.expert_num; i++) {
      m_mem_requests.push_back(
          {(void **)&gate_up_ba_ptr[i], T::BufferA::required_size(config_.max_len, config_.hidden_size)});
      m_mem_requests.push_back(
          {(void **)&gate_bc_ptr[i], T::BufferC::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests.push_back(
          {(void **)&up_bc_ptr[i], T::BufferC::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests.push_back(
          {(void **)&down_ba_ptr[i], T::BufferA::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests.push_back(
          {(void **)&down_bc_ptr[i], T::BufferC::required_size(config_.max_len, config_.hidden_size)});
    }
    shared_mem_buffer.alloc(this, m_mem_requests);

    m_local_pos_.resize(config_.max_len);
    for (int i = 0; i < config_.max_len; i++) {
      m_local_pos_[i].resize(config_.routed_expert_num);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    m_local_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

	m_local_down_grad_ptr_.resize(config_.expert_num);
    m_local_v_grad_ptr_.resize(config_.expert_num);
    m_local_gate_grad_ptr_fp32_.resize(config_.expert_num);
    m_local_up_grad_ptr_fp32_.resize(config_.expert_num);
    m_local_gate_grad_ptr_bf16_.resize(config_.expert_num);
    m_local_up_grad_ptr_bf16_.resize(config_.expert_num);
    m_local_gate_in_grad_ptr_.resize(config_.expert_num);
    m_local_up_in_grad_ptr_.resize(config_.expert_num);

    for (uint64_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, gate_up_ba_ptr[i]));
      gate_bc_.push_back(
          std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, gate_bc_ptr[i]));
      up_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, up_bc_ptr[i]));
      down_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, down_ba_ptr[i]));
      down_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, down_bc_ptr[i]));

#ifdef USE_NUMA
      int numa_nodes = numa_num_configured_nodes();
      gate_bb_numa_.resize(numa_nodes);
      up_bb_numa_.resize(numa_nodes);
      down_bb_numa_.resize(numa_nodes);
      for (int j = 0; j < numa_nodes; j++) {
        void *gate_bb_ptr =
            numa_alloc_aligned(T::BufferB::required_size(config_.intermediate_size, config_.hidden_size), j, 64);
        gate_bb_numa_[j].push_back(
            std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, gate_bb_ptr));
        void *up_bb_ptr =
            numa_alloc_aligned(T::BufferB::required_size(config_.intermediate_size, config_.hidden_size), j, 64);
        up_bb_numa_[j].push_back(
            std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, up_bb_ptr));
        void *down_bb_ptr =
            numa_alloc_aligned(T::BufferB::required_size(config_.hidden_size, config_.intermediate_size), j, 64);
        down_bb_numa_[j].push_back(  
            std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, down_bb_ptr));
      }
#else
      void *gate_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size));
      gate_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, gate_bb_ptr));

      void *up_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size));
      up_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, up_bb_ptr));

      void *down_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
      down_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, down_bb_ptr));
#endif

	// 转置权重 buffer (bb_t)
      void *gate_bb_ptr_t = std::aligned_alloc(
          64, T::BufferB::required_size(config_.hidden_size,
                                        config_.intermediate_size));
      gate_bb_t_.push_back(std::make_shared<typename T::BufferB>(
          config_.hidden_size, config_.intermediate_size, gate_bb_ptr_t));

      void *up_bb_ptr_t = std::aligned_alloc(
          64, T::BufferB::required_size(config_.hidden_size,
                                        config_.intermediate_size));
      up_bb_t_.push_back(std::make_shared<typename T::BufferB>(
          config_.hidden_size, config_.intermediate_size, up_bb_ptr_t));

      void *down_bb_ptr_t = std::aligned_alloc(
          64, T::BufferB::required_size(config_.intermediate_size,
                                        config_.hidden_size));
      down_bb_t_.push_back(std::make_shared<typename T::BufferB>(
          config_.intermediate_size, config_.hidden_size, down_bb_ptr_t));
    }
	
	m_forward_cache_.resize(config_.max_len);
    for (auto &row : m_forward_cache_)
      row.init(config_.intermediate_size);
  }

  ~SFT_AMX_MOE() { shared_mem_buffer.dealloc(this); }

  void load_weights(Backend *backend) {
    int nth = T::recommended_nth(config_.intermediate_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;
#ifdef USE_NUMA
          int numa_nodes = numa_num_configured_nodes();
          for (int j = 0; j < numa_nodes; j++) {
            gate_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)config_.gate_proj +
                                                       expert_idx * config_.intermediate_size * config_.hidden_size,
                                                   ith, nth);
            up_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)config_.up_proj +
                                                     expert_idx * config_.intermediate_size * config_.hidden_size,
                                                 ith, nth);
          }
#else
          gate_bb_[expert_idx]->from_mat((ggml_bf16_t *)config_.gate_proj +
                                             expert_idx * config_.intermediate_size * config_.hidden_size,
                                         ith, nth);
          up_bb_[expert_idx]->from_mat(
              (ggml_bf16_t *)config_.up_proj + expert_idx * config_.intermediate_size * config_.hidden_size, ith, nth);
#endif
		// 转置权重
		  gate_bb_t_[expert_idx]->from_mat_transpose_backward(
			  (ggml_bf16_t *)config_.gate_proj + expert_idx * config_.intermediate_size * config_.hidden_size, ith, nth);
          up_bb_t_[expert_idx]->from_mat_transpose_backward(
              (ggml_bf16_t *)config_.up_proj + expert_idx * config_.intermediate_size * config_.hidden_size, ith, nth);
        },
        nullptr);
    nth = T::recommended_nth(config_.hidden_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;
#ifdef USE_NUMA
          int numa_nodes = numa_num_configured_nodes();
          for (int j = 0; j < numa_nodes; j++) {
            down_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)config_.down_proj +
                                                       expert_idx * config_.hidden_size * config_.intermediate_size,
                                                   ith, nth);
          }
#else
          down_bb_[expert_idx]->from_mat((ggml_bf16_t *)config_.down_proj +
                                             expert_idx * config_.hidden_size * config_.intermediate_size,
                                         ith, nth);
#endif
		  down_bb_t_[expert_idx]->from_mat_transpose_backward(
              (ggml_bf16_t *)config_.down_proj +
                  expert_idx * config_.hidden_size * config_.intermediate_size,
              ith, nth);
        },
        nullptr);
  }

  void warm_up(Backend *backend) {}

  void forward(int qlen, int k, const uint64_t *expert_ids, const float *weights, const void *input, void *output,
               int *batch_size_tensor, Backend *backend) {
    qlen = batch_size_tensor[0];
    bool use_amx = (qlen > 4 * config_.expert_num / config_.routed_expert_num);
    int activated_expert = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }
    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
    }
    backend->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int j = 0; j < k; j++) {
            memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
                   (ggml_bf16_t *)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
          }
        },
        nullptr);
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];

          gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
        },
        nullptr);
    int nth = T::recommended_nth(config_.intermediate_size);
    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
#ifdef USE_NUMA
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                       gate_up_ba_[expert_idx], gate_bb_numa_[Backend::numa_node][expert_idx], gate_bc_[expert_idx],
                       ith, nth, use_amx);
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                       gate_up_ba_[expert_idx], up_bb_numa_[Backend::numa_node][expert_idx], up_bc_[expert_idx], ith,
                       nth, use_amx);
#else
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                       gate_up_ba_[expert_idx], gate_bb_[expert_idx], gate_bc_[expert_idx], ith, nth, use_amx);
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                       gate_up_ba_[expert_idx], up_bb_[expert_idx], up_bc_[expert_idx], ith, nth, use_amx);
#endif
          gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
          for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            ggml_bf16_t *gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
            ggml_bf16_t *up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
            for (int j = n_start; j < n_end; j += 32) {
              __m512 gate_val0, gate_val1, up_val0, up_val1;
              avx512_32xbf16_to_32xfp32((__m512i *)(gate_output_ptr + j), &gate_val0, &gate_val1);
              avx512_32xbf16_to_32xfp32((__m512i *)(up_output_ptr + j), &up_val0, &up_val1);
              __m512 result0 = sft_act_fn(gate_val0, up_val0);
              __m512 result1 = sft_act_fn(gate_val1, up_val1);
              avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i *)(gate_output_ptr + j));
            }
          }
        },
        nullptr);
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);
    nth = T::recommended_nth(config_.hidden_size);
    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
#ifdef USE_NUMA
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx],
                       down_bb_numa_[Backend::numa_node][expert_idx], down_bc_[expert_idx], ith, nth, use_amx);
#else
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx],
                       down_bb_[expert_idx], down_bc_[expert_idx], ith, nth, use_amx);
#endif
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);
    backend->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              __m512 weight = _mm512_set1_ps(weights[i * k + j]);
              __m512 down_output0, down_output1;
              avx512_32xbf16_to_32xfp32((__m512i *)(m_local_down_output_ptr_[expert_ids[i * k + j]] +
                                                    m_local_pos_[i][j] * config_.hidden_size + e),
                                        &down_output0, &down_output1);
              x0 = _mm512_fmadd_ps(down_output0, weight, x0);
              x1 = _mm512_fmadd_ps(down_output1, weight, x1);
            }
            avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i *)((ggml_bf16_t *)output + i * config_.hidden_size + e));
          }
        },
        nullptr);
  }

  void backward(int qlen, int k, const uint64_t *expert_ids,
                const float *weights, const void *grad_output,
                void *grad_input, Backend *backend) {

    /* ---------- STEP-0: Routing for grad_output ---------- */
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int t = 0; t < qlen; ++t)
      for (int j = 0; j < k; ++j)
        m_local_pos_[t][j] = m_local_num_[expert_ids[t * k + j]]++;

    size_t offset = 0;
    for (int e = 0; e < config_.expert_num; ++e) {
      m_local_down_grad_ptr_[e] =
          m_local_down_grad_ + offset * config_.hidden_size;
      m_local_v_grad_ptr_[e] =
          m_local_v_grad_fp32_ + offset * config_.intermediate_size;
      m_local_gate_grad_ptr_fp32_[e] =
          m_local_gate_grad_fp32_ + offset * config_.intermediate_size;
      m_local_up_grad_ptr_fp32_[e] =
          m_local_up_grad_fp32_ + offset * config_.intermediate_size;
      m_local_gate_grad_ptr_bf16_[e] =
          m_local_gate_grad_bf16_ + offset * config_.intermediate_size;
      m_local_up_grad_ptr_bf16_[e] =
          m_local_up_grad_bf16_ + offset * config_.intermediate_size;
      m_local_gate_in_grad_ptr_[e] =
          m_local_gate_in_grad_fp32_ + offset * config_.hidden_size;
      m_local_up_in_grad_ptr_[e] =
          m_local_up_in_grad_fp32_ + offset * config_.hidden_size;
      offset += m_local_num_[e];
    }

    backend->do_work_stealing_job(
        qlen, nullptr,
        [&](int t) {
          const ggml_bf16_t *src =
              (const ggml_bf16_t *)grad_output + t * config_.hidden_size;
          for (int j = 0; j < k; ++j) {
            uint64_t eid = expert_ids[t * k + j];
            memcpy(m_local_down_grad_ptr_[eid] +
                       m_local_pos_[t][j] * config_.hidden_size,
                   src, sizeof(ggml_bf16_t) * config_.hidden_size);
          }
        },
        nullptr);

    /* ---------- STEP-1: V_grad = Down^T · grad_output ---------- */
    int nth = T::recommended_nth(config_.intermediate_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, [&]([[maybe_unused]] int) { T::config(); },
        [&](int task_id) {
          int expert_idx = task_id / nth;
          int ith = task_id % nth;

          /* pack A (grad_output) */
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx],
                                         m_local_down_grad_ptr_[expert_idx], 0,
                                         1);

          amx::mat_mul(
              m_local_num_[expert_idx], config_.intermediate_size,
              config_.hidden_size, down_ba_[expert_idx],
              down_bb_t_[expert_idx], gate_bc_[expert_idx], ith, nth,
              /*use_amx=*/true);

          /* 写回 m_local_v_grad_fp32_ (FP32) */
          gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx],
                                       (ggml_bf16_t
                                            *)(m_local_v_grad_ptr_[expert_idx]),
                                       ith, nth);
        },
        nullptr);

    /* ---------- STEP-2: gate/up grad & act grad ---------- */
    backend->do_work_stealing_job(
        config_.expert_num, nullptr,
        [&](int expert_idx) {
          int rows = m_local_num_[expert_idx];
          for (int row = 0; row < rows; ++row) {
            int token_idx   = 0; // DEMO: 若有 token-expert 映射表可替换
            int routed_slot = 0;
            float *v_grad =
                m_local_v_grad_ptr_[expert_idx] + row * config_.intermediate_size;
            float *g_grad =
                m_local_gate_grad_ptr_fp32_[expert_idx] +
                row * config_.intermediate_size;
            float *u_grad =
                m_local_up_grad_ptr_fp32_[expert_idx] +
                row * config_.intermediate_size;
            const float *g_cache =
                m_forward_cache_[token_idx].gate_u.data();
            const float *u_cache =
                m_forward_cache_[token_idx].up_v.data();

            for (int col = 0; col < config_.intermediate_size; col += 32) {
              __m512 v0 = _mm512_loadu_ps(v_grad + col);
              __m512 v1 = _mm512_loadu_ps(v_grad + col + 16);

              __m512 g0 = _mm512_loadu_ps(g_cache + col);
              __m512 g1 = _mm512_loadu_ps(g_cache + col + 16);
              __m512 u0 = _mm512_loadu_ps(u_cache + col);
              __m512 u1 = _mm512_loadu_ps(u_cache + col + 16);

              __m512 actg0 = sft_act_fn_grad(g0);
              __m512 actg1 = sft_act_fn_grad(g1);
              __m512 swi0  = _mm512_mul_ps(sft_sigmoid(g0), g0);
              __m512 swi1  = _mm512_mul_ps(sft_sigmoid(g1), g1);

              __m512 g_grad0 = _mm512_mul_ps(_mm512_mul_ps(v0, u0), actg0);
              __m512 u_grad0 = _mm512_mul_ps(v0, swi0);
              __m512 g_grad1 = _mm512_mul_ps(_mm512_mul_ps(v1, u1), actg1);
              __m512 u_grad1 = _mm512_mul_ps(v1, swi1);

              _mm512_storeu_ps(g_grad + col, g_grad0);
              _mm512_storeu_ps(g_grad + col + 16, g_grad1);
              _mm512_storeu_ps(u_grad + col, u_grad0);
              _mm512_storeu_ps(u_grad + col + 16, u_grad1);
            }
          }
          /* 压 BF16 */
          gate_up_ba_[expert_idx]->from_mat(
              m_local_num_[expert_idx],
              (ggml_bf16_t *)(m_local_gate_grad_ptr_fp32_[expert_idx]), 0, 1);
          down_ba_[expert_idx]->from_mat(
              m_local_num_[expert_idx],
              (ggml_bf16_t *)(m_local_up_grad_ptr_fp32_[expert_idx]), 0, 1);
        },
        nullptr);

    /* ---------- STEP-3: 输入梯度 = G^T·ggrad + U^T·ugrad ---------- */
    nth = T::recommended_nth(config_.hidden_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, [&]([[maybe_unused]] int) { T::config(); },
        [&](int task_id) {
          int expert_idx = task_id / nth;
          int ith = task_id % nth;

          amx::mat_mul(
              m_local_num_[expert_idx], config_.hidden_size,
              config_.intermediate_size, gate_up_ba_[expert_idx],
              gate_bb_t_[expert_idx], gate_bc_[expert_idx], ith, nth,
              /*use_amx=*/true);
          amx::mat_mul(
              m_local_num_[expert_idx], config_.hidden_size,
              config_.intermediate_size, down_ba_[expert_idx],
              up_bb_t_[expert_idx], down_bc_[expert_idx], ith, nth,
              /*use_amx=*/true);

          /* 累加两个 BufferC → gate_in_grad_ptr */
          float *g_in =
              m_local_gate_in_grad_ptr_[expert_idx] + ith * T::M_STEP;
          float *tmp_g =
              gate_bc_[expert_idx]
                  ->get_submat(m_local_num_[expert_idx], config_.hidden_size,
                               ith * T::M_STEP, 0);
          float *tmp_u =
              down_bc_[expert_idx]
                  ->get_submat(m_local_num_[expert_idx], config_.hidden_size,
                               ith * T::M_STEP, 0);
          int rows = std::min(T::M_STEP, m_local_num_[expert_idx] - ith * T::M_STEP);
          for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < config_.hidden_size; ++c) {
              g_in[r * config_.hidden_size + c] =
                  tmp_g[r * config_.hidden_size + c] +
                  tmp_u[r * config_.hidden_size + c];
            }
          }
        },
        nullptr);

    /* ---------- STEP-4: scatter + 写回 grad_input (BF16) ---------- */
    backend->do_work_stealing_job(
        qlen, nullptr,
        [&](int t) {
          float tmp32[32];
          for (int col = 0; col < config_.hidden_size; col += 32) {
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            for (int j = 0; j < k; ++j) {
              uint64_t eid = expert_ids[t * k + j];
              __m512 v0, v1;
              const float *src = m_local_gate_in_grad_ptr_[eid] +
                                 m_local_pos_[t][j] * config_.hidden_size + col;
              v0 = _mm512_loadu_ps(src);
              v1 = _mm512_loadu_ps(src + 16);
              sum0 = _mm512_add_ps(sum0, v0);
              sum1 = _mm512_add_ps(sum1, v1);
            }
            avx512_32xfp32_to_32xbf16(
                &sum0, &sum1,
                (__m512i *)((ggml_bf16_t *)grad_input + t * config_.hidden_size +
                            col));
          }
        },
        nullptr);
  }
};
#endif
