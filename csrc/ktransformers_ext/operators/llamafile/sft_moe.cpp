/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-15 07:43:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "sft_moe.h"
#include "debug_sft_moe.hpp"
#include <iostream>
#include <cstdint>
#include <cstring>
#include <time.h>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

inline void dump_grad_bin(const std::string &file_name,
                          const void       *data,
                          size_t            elem_cnt,
                          ggml_type         dtype,
						  std::streamoff    offset_bytes = 0)
{
    std::string path = get_env_or_default("SFT_DEBUG_PATH","debug") + "/" + file_name;
    switch (dtype) {
        case GGML_TYPE_F32:  path += ".f32";  break;
        case GGML_TYPE_F16:  path += ".f16";  break;
        case GGML_TYPE_BF16: path += ".bf16"; break;
        default:             path += ".raw";  break;
    }
	std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f.is_open()) {
        std::ofstream tmp(path, std::ios::out | std::ios::binary);
        tmp.close();
        f.open(path, std::ios::in | std::ios::out | std::ios::binary);
    }

    f.seekp(offset_bytes * ggml_type_size(dtype));
	// std::cout << "seekp: " << offset_bytes * ggml_type_size(dtype) << std::endl;

    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(elem_cnt * ggml_type_size(dtype)));
    f.close();
}

SFT_MOE::SFT_MOE(SFT_MOEConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;
    
    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    gate_proj_numa_.resize(numa_nodes);
    up_proj_numa_.resize(numa_nodes);
    down_proj_numa_.resize(numa_nodes);
    size_t exp_inter_hidden_mul_ = (size_t)config.expert_num * config.intermediate_size * config.hidden_size;
    for (int i = 0; i < numa_nodes; i++) {
        gate_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type), i);
        up_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type), i);
        down_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type), i);
        if (!gate_proj_numa_[i]) {
            std::cout << "Memory allocation failed for gate_proj_numa_ on node " << i << std::endl;
        }
        if (!up_proj_numa_[i]) {
            std::cout << "Memory allocation failed for up_proj_numa_ on node " << i << std::endl;
        }
        if (!down_proj_numa_[i]) {
            std::cout << "Memory allocation failed for down_proj_numa_ on node " << i << std::endl;
        }
        memcpy(gate_proj_numa_[i], gate_proj_, exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type));
        memcpy(up_proj_numa_[i], up_proj_, exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type));
        memcpy(down_proj_numa_[i], down_proj_, exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type));
    }
    #endif

    gate_proj_t_ = (uint8_t*)aligned_alloc(64, config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.grad_type));
    up_proj_t_ = (uint8_t*)aligned_alloc(64, config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.grad_type));
    down_proj_t_ = (uint8_t*)aligned_alloc(64, config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.grad_type));
    transpose_buffer_fp32_ = (float*)aligned_alloc(64, config_.expert_num * config_.intermediate_size * config_.hidden_size * sizeof(float));
    transpose_buffer_ = (uint8_t*)aligned_alloc(64, config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.grad_type));
    
    std::vector<std::pair<void**, uint64_t>> s_mem_requests;
    s_mem_requests.push_back({(void**)&s_input_fp32_, sizeof(float) * config_.hidden_size});
    s_mem_requests.push_back({(void**)&s_gate_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
    s_mem_requests.push_back({(void**)&s_up_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    s_gate_output_.resize(config_.routed_expert_num);
    s_up_output_.resize(config_.routed_expert_num);
    s_intermediate_fp32_.resize(config_.routed_expert_num);
    s_down_input_.resize(config_.routed_expert_num);
    s_down_output_.resize(config_.routed_expert_num);
    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_mem_requests.push_back({(void**)&s_gate_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_up_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_down_input_[i], config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type)});
        s_mem_requests.push_back({(void**)&s_down_output_[i], sizeof(float) * config_.hidden_size});
    }
    s_mem_requests.push_back({(void**)&s_output_fp32_, sizeof(float) * config_.hidden_size});
        
    s_down_input_grad_.resize(config_.routed_expert_num);
    s_gate_output_grad_fp32_.resize(config_.routed_expert_num);
    s_up_output_grad_fp32_.resize(config_.routed_expert_num);
    s_gate_output_grad_.resize(config_.routed_expert_num);
    s_up_output_grad_.resize(config_.routed_expert_num);
    s_gate_input_grad_.resize(config_.routed_expert_num);
    s_up_input_grad_.resize(config_.routed_expert_num);
    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_mem_requests.push_back({(void**)&s_down_input_grad_[i], config_.intermediate_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_gate_output_grad_fp32_[i], config_.intermediate_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_up_output_grad_fp32_[i], config_.intermediate_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_gate_output_grad_[i], config_.intermediate_size * ggml_type_size(config_.grad_type)});
        s_mem_requests.push_back({(void**)&s_up_output_grad_[i], config_.intermediate_size * ggml_type_size(config_.grad_type)});
        s_mem_requests.push_back({(void**)&s_gate_input_grad_[i], config_.hidden_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_up_input_grad_[i], config_.hidden_size * sizeof(float)});
    }
    s_mem_requests.push_back({(void**)&s_input_grad_fp32_, config_.hidden_size * sizeof(float)});

    shared_mem_buffer.alloc(this, s_mem_requests);

    std::vector<std::pair<void**, uint64_t>> m_mem_requests;
    m_input_fp32_.resize(config_.group_max_len);
    m_gate_input_.resize(config_.group_max_len);
    m_up_input_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_input_fp32_[i], sizeof(float) * config_.hidden_size});
        m_mem_requests.push_back({(void**)&m_gate_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
        m_mem_requests.push_back({(void**)&m_up_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    }
    m_mem_requests.push_back({(void**)&m_local_gate_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_up_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_gate_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_up_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_intermediate_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_down_input_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_down_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_output_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_output_fp32_[i], sizeof(float) * config_.hidden_size});
    }
    
    m_mem_requests.push_back({(void**)&m_local_down_output_grad_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(config_.grad_type)});
    m_mem_requests.push_back({(void**)&m_local_down_input_grad_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_gate_output_grad_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_up_output_grad_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_gate_output_grad_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    m_mem_requests.push_back({(void**)&m_local_up_output_grad_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    m_mem_requests.push_back({(void**)&m_local_gate_input_grad_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_mem_requests.push_back({(void**)&m_local_up_input_grad_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_mem_requests.push_back({(void**)&m_local_token_indices_, sizeof(int) * config_.routed_expert_num * config_.group_max_len});
    m_mem_requests.push_back({(void**)&m_local_expert_positions_, sizeof(int) * config_.routed_expert_num * config_.group_max_len});
    m_grad_input_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_grad_input_fp32_[i], sizeof(float) * config_.hidden_size});
    }
    
    shared_mem_buffer.alloc(this, m_mem_requests);

    m_local_pos_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_local_pos_[i].resize(config_.routed_expert_num);
    }
    m_local_num_.resize(config_.expert_num);
    m_local_gate_input_ptr_.resize(config_.expert_num);
    m_local_up_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_intermediate_fp32_ptr_.resize(config_.expert_num);
    m_local_down_input_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);
    
    m_local_down_output_grad_ptr_.resize(config_.expert_num);
    m_local_down_input_grad_ptr_.resize(config_.expert_num);
    m_local_gate_output_grad_fp32_ptr_.resize(config_.expert_num);
    m_local_up_output_grad_fp32_ptr_.resize(config_.expert_num);
    m_local_gate_output_grad_ptr_.resize(config_.expert_num);
    m_local_up_output_grad_ptr_.resize(config_.expert_num);
    m_local_gate_input_grad_ptr_.resize(config_.expert_num);
    m_local_up_input_grad_ptr_.resize(config_.expert_num);
    
    m_local_token_indices_ptr_.resize(config_.expert_num);
    m_local_expert_positions_ptr_.resize(config_.expert_num);
}

SFT_MOE::~SFT_MOE() {
    shared_mem_buffer.dealloc(this);

    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    for (int i = 0; i < numa_nodes; i++) {
        numa_free(gate_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type));
        numa_free(up_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type));
        numa_free(down_proj_numa_[i], config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type));
    }
    #endif
}

void SFT_MOE::warm_up(Backend* backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}

static float act_fn_grad(float x) {
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    return sigmoid_x * (1. + x * (1. - sigmoid_x));
}

void SFT_MOE::forward_one(int layer_idx, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
            gate_input_ptr = up_input_ptr = s_gate_input_;
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = s_gate_input_;
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(s_input_fp32_, s_up_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = s_up_input_;
            } else {
                up_input_ptr = input;
            }
        }
    }
    int nth = config_.intermediate_size / config_.stride;
    backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id % nth;
        
        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif

        float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
        }
        if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
            float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx] + ith * config_.stride;
            void* down_input_ptr = s_down_input_[expert_idx] + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        for (int i = 0; i < k; i++) {
            from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }
    nth = config_.hidden_size / config_.stride;
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_output_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];

            #ifdef USE_NUMA
            void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #else
            void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #endif
            
            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_output_fp32_[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            float* output_fp32_ptr = s_output_fp32_ + ith * config_.stride;
            void* output_ptr = (uint8_t*)output + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
            from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
    }, nullptr);
}

void SFT_MOE::forward_many(int layer_idx, int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
        m_local_down_input_ptr_[i] = m_local_down_input_ + offset * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
        offset += m_local_num_[i];
    }

    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        const void* gate_input_ptr;
        const void* up_input_ptr;
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), gate_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), up_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        }
    }, nullptr);
    int stride = QK_K;
    int nth = config_.intermediate_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif

        float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) * m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
            }
            float* intermediate_fp32_ptr = m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* down_input_ptr = m_local_down_input_ptr_[expert_idx] + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    stride = QK_K;
    nth = config_.hidden_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* down_input_ptr = m_local_down_input_ptr_[expert_idx];
        
        #ifdef USE_NUMA
        void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #else
        void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #endif

        float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] * weights[i * k + j];
            }
        }
        from_float(m_output_fp32_[i], (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    }, nullptr);
}

void SFT_MOE::forward(int layer_idx, int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
	// for (uint64_t eid = 0; eid < (uint64_t)config_.expert_num; ++eid) {
    //     int rows = m_local_num_[eid];
    //     if (rows == 0) continue;

    //     const int H = config_.hidden_size;
    //     const int I = config_.intermediate_size;
    //     size_t rowsH = (size_t)rows * H;
    //     size_t rowsI = (size_t)rows * I;

	// 	dump_grad_bin("layer0_E_Forward"+std::to_string(eid)+"_up_proj", (uint8_t*)up_proj_ + eid * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.up_type), (size_t)config_.hidden_size * config_.intermediate_size, config_.up_type);

	// 	dump_grad_bin("layer0_E_Forward"+std::to_string(eid)+"_gate_proj", (uint8_t*)gate_proj_ + eid * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.gate_type), (size_t)config_.hidden_size * config_.intermediate_size, config_.gate_type);

	// 	dump_grad_bin("layer0_E_Forward"+std::to_string(eid)+"_down_proj", (uint8_t*)down_proj_ + eid * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.down_type), (size_t)config_.hidden_size * config_.intermediate_size, config_.down_type);
	// }

	std::cout << "[forward] layer_idx: " << layer_idx << std::endl;
	std::cout << "config_.routed_expert_num: " << config_.routed_expert_num << std::endl;
	std::cout << "config_.expert_num: " << config_.expert_num << std::endl;

    if (qlen < config_.group_min_len) {
        for (int i = 0; i < qlen; i++) {
            forward_one(layer_idx, k, expert_ids + i * k, weights + i * k, (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
        }
        return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);
    forward_many(layer_idx, forward_len, k, expert_ids, weights, input, output, backend);
    forward(layer_idx, qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k, (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
}

void SFT_MOE::transpose_expert(const void* src, void* dst, int R, int C, ggml_type src_type, ggml_type dst_type, uint64_t expert_idx) {
    to_float(src, transpose_buffer_fp32_ + (R * C * expert_idx), R * C, src_type);
    from_float(transpose_buffer_fp32_ + (R * C * expert_idx), (uint8_t*)transpose_buffer_ + (R * C * expert_idx) * ggml_type_size(dst_type), R * C, dst_type);
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            memcpy(
                (uint8_t*)dst + (c * R + r) * ggml_type_size(dst_type),
                (uint8_t*)transpose_buffer_ + (R * C * expert_idx + r * C + c) * ggml_type_size(dst_type),
                ggml_type_size(dst_type));
        }
    }
}

void SFT_MOE::get_transpose(Backend* backend) {
    // Transpose gate_proj_
    int R_gate = config_.intermediate_size;
    int C_gate = config_.hidden_size;
    backend->do_work_stealing_job(config_.expert_num, nullptr, [&](int expert_idx) {
        void* src_expert = (uint8_t*)gate_proj_ + expert_idx * (size_t)R_gate * C_gate * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        void* dst_expert_t = (uint8_t*)gate_proj_t_ + expert_idx * (size_t)C_gate * R_gate * ggml_type_size(config_.grad_type);
        transpose_expert(src_expert, dst_expert_t, R_gate, C_gate, config_.gate_type, config_.grad_type, expert_idx);
	}, nullptr);
    // Transpose up_proj_
    int R_up = config_.intermediate_size;
    int C_up = config_.hidden_size;
    backend->do_work_stealing_job(config_.expert_num, nullptr, [&](int expert_idx) {
        void* src_expert = (uint8_t*)up_proj_ + expert_idx * (size_t)R_up * C_up * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        void* dst_expert_t = (uint8_t*)up_proj_t_ + expert_idx * (size_t)C_up * R_up * ggml_type_size(config_.grad_type);
        transpose_expert(src_expert, dst_expert_t, R_up, C_up, config_.up_type, config_.grad_type, expert_idx);
	}, nullptr);
    // Transpose down_proj_
    int R_down = config_.hidden_size;
    int C_down = config_.intermediate_size;
    backend->do_work_stealing_job(config_.expert_num, nullptr, [&](int expert_idx) {
        void* src_expert = (uint8_t*)down_proj_ + expert_idx * (size_t)R_down * C_down * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        void* dst_expert_t = (uint8_t*)down_proj_t_ + expert_idx * (size_t)C_down * R_down * ggml_type_size(config_.grad_type);
        transpose_expert(src_expert, dst_expert_t, R_down, C_down, config_.down_type, config_.grad_type, expert_idx);
	}, nullptr);
}

void SFT_MOE::backward_one(int layer_idx, int k, const uint64_t* expert_ids, const float* weights, const void* input, const void* output_grad, void* input_grad, Backend* backend) {
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
            gate_input_ptr = up_input_ptr = s_gate_input_;
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = s_gate_input_;
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(s_input_fp32_, s_up_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = s_up_input_;
            } else {
                up_input_ptr = input;
            }
        }
    }

    int nth = config_.intermediate_size / config_.stride;
    backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id % nth;

        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif
        float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif
        float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

		// clkz1 = clock();
        void* down_proj_t_ptr = (uint8_t*)down_proj_t_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.grad_type);
        float* down_input_grad_ptr = s_down_input_grad_[expert_idx] + ith * config_.stride;
        // clkz2 = clock();
        llamafile_sgemm(config_.stride, 1, config_.hidden_size, down_proj_t_ptr, config_.hidden_size, output_grad, config_.hidden_size, down_input_grad_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, config_.grad_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        // clkz3 = clock();

		for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; ++i) {
			s_down_input_grad_[expert_idx][i] *= weights[expert_idx];
			s_gate_output_grad_fp32_[expert_idx][i] = s_down_input_grad_[expert_idx][i] * s_up_output_[expert_idx][i] * act_fn_grad(s_gate_output_[expert_idx][i]);
			s_up_output_grad_fp32_[expert_idx][i] = s_down_input_grad_[expert_idx][i] * act_fn(s_gate_output_[expert_idx][i]);
		}
        // clkz4 = clock();
        from_float(s_gate_output_grad_fp32_[expert_idx] + ith * config_.stride, s_gate_output_grad_[expert_idx] + ith * config_.stride * ggml_type_size(config_.grad_type), config_.stride, config_.grad_type);
        from_float(s_up_output_grad_fp32_[expert_idx] + ith * config_.stride, s_up_output_grad_[expert_idx] + ith * config_.stride * ggml_type_size(config_.grad_type), config_.stride, config_.grad_type);
        // clkz5 = clock();
    }, nullptr);

	// clk3 = clock();
    nth = config_.hidden_size / config_.stride;
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_input_grad_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];

            void* gate_proj_t_ptr = (uint8_t*)gate_proj_t_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
            float* gate_input_grad_ptr = s_gate_input_grad_[expert_idx] + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size, gate_proj_t_ptr, config_.intermediate_size, s_gate_output_grad_[expert_idx], config_.intermediate_size, gate_input_grad_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, ggml_internal_get_type_traits(config_.grad_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

            void* up_proj_t_ptr = (uint8_t*)up_proj_t_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
            float* up_input_grad_ptr = s_up_input_grad_[expert_idx] + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size, up_proj_t_ptr, config_.intermediate_size, s_up_output_grad_[expert_idx], config_.intermediate_size, up_input_grad_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, config_.grad_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_input_grad_fp32_[i] += s_gate_input_grad_[expert_idx][i] + s_up_input_grad_[expert_idx][i];
            }
        }
        from_float(s_input_grad_fp32_ + ith * config_.stride, (uint8_t*)input_grad + ith * config_.stride * ggml_type_size(config_.grad_type), config_.stride, config_.grad_type);
    }, nullptr);
}

void SFT_MOE::backward_many(int layer_idx, int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, const void* output_grad, void* input_grad, Backend* backend) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        
        m_local_down_output_grad_ptr_[i] = m_local_down_output_grad_ + offset * config_.hidden_size * ggml_type_size(config_.grad_type);
        m_local_down_input_grad_ptr_[i] = m_local_down_input_grad_ + offset * config_.intermediate_size;
        m_local_gate_output_grad_fp32_ptr_[i] = m_local_gate_output_grad_fp32_ + offset * config_.intermediate_size;
        m_local_up_output_grad_fp32_ptr_[i] = m_local_up_output_grad_fp32_ + offset * config_.intermediate_size;
        m_local_gate_output_grad_ptr_[i] = m_local_gate_output_grad_ + offset * config_.intermediate_size * ggml_type_size(config_.grad_type);
        m_local_up_output_grad_ptr_[i] = m_local_up_output_grad_ + offset * config_.intermediate_size * ggml_type_size(config_.grad_type);
        m_local_gate_input_grad_ptr_[i] = m_local_gate_input_grad_ + offset * config_.hidden_size;
        m_local_up_input_grad_ptr_[i] = m_local_up_input_grad_ + offset * config_.hidden_size;
        m_local_token_indices_ptr_[i] = m_local_token_indices_ + offset;
        m_local_expert_positions_ptr_[i] = m_local_expert_positions_ + offset;
        offset += m_local_num_[i];
    }

    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        const void* gate_input_ptr;
        const void* up_input_ptr;
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), gate_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), up_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
            
            uint64_t expert_id = expert_ids[i * k + j];
            int local_row = m_local_pos_[i][j];
            memcpy(m_local_down_output_grad_ptr_[expert_id] + local_row * config_.hidden_size * ggml_type_size(config_.grad_type), (uint8_t*)output_grad + i * config_.hidden_size * ggml_type_size(config_.grad_type), config_.hidden_size * ggml_type_size(config_.grad_type));
            m_local_token_indices_ptr_[expert_id][local_row] = i;
            m_local_expert_positions_ptr_[expert_id][local_row] = j;
        }
    }, nullptr);

    int stride = QK_K;
    int nth = config_.intermediate_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;

        void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];
        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif
        float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        void* up_input_ptr = m_local_up_input_ptr_[expert_idx];
        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif
        float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        void* down_proj_t_ptr = (uint8_t*)down_proj_t_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.grad_type);
        void* down_output_grad_ptr = m_local_down_output_grad_ptr_[expert_idx];
        float* down_input_grad_ptr = m_local_down_input_grad_ptr_[expert_idx] + ith * stride;
                    
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size, down_proj_t_ptr, config_.hidden_size, down_output_grad_ptr, config_.hidden_size, down_input_grad_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, config_.grad_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            int token_idx = m_local_token_indices_ptr_[expert_idx][i];
            int expert_pos = m_local_expert_positions_ptr_[expert_idx][i];
            float weight = weights[token_idx * k + expert_pos];
            
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                m_local_down_input_grad_ptr_[expert_idx][i * config_.intermediate_size + j] *= weight;
                
                float down_input_grad = m_local_down_input_grad_ptr_[expert_idx][i * config_.intermediate_size + j];
                m_local_gate_output_grad_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = down_input_grad * m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j] * act_fn_grad(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]);
                m_local_up_output_grad_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = down_input_grad * act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]);
            }
            
            float* gate_output_grad_fp32_ptr = m_local_gate_output_grad_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* gate_output_grad_ptr = m_local_gate_output_grad_ptr_[expert_idx] + (i * config_.intermediate_size + ith * stride) * ggml_type_size(config_.grad_type);
            from_float(gate_output_grad_fp32_ptr, gate_output_grad_ptr, stride, config_.grad_type);
            
            float* up_output_grad_fp32_ptr = m_local_up_output_grad_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* up_output_grad_ptr = m_local_up_output_grad_ptr_[expert_idx] + (i * config_.intermediate_size + ith * stride) * ggml_type_size(config_.grad_type);
            from_float(up_output_grad_fp32_ptr, up_output_grad_ptr, stride, config_.grad_type);
        }
    }, nullptr); 
    stride = QK_K;
    nth = config_.hidden_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        
        void* gate_proj_t_ptr = (uint8_t*)gate_proj_t_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
        void* up_proj_t_ptr = (uint8_t*)up_proj_t_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
        void* gate_output_grad_ptr = m_local_gate_output_grad_ptr_[expert_idx];
        void* up_output_grad_ptr = m_local_up_output_grad_ptr_[expert_idx];
        float* gate_input_grad_ptr = m_local_gate_input_grad_ptr_[expert_idx] + ith * stride;
        float* up_input_grad_ptr = m_local_up_input_grad_ptr_[expert_idx] + ith * stride;

        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size, gate_proj_t_ptr, config_.intermediate_size, gate_output_grad_ptr, config_.intermediate_size, gate_input_grad_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, config_.grad_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size, up_proj_t_ptr, config_.intermediate_size, up_output_grad_ptr, config_.intermediate_size, up_input_grad_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, config_.grad_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_grad_input_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_grad_input_fp32_[i][e] += m_local_gate_input_grad_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] + m_local_up_input_grad_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e];
            }
        }
        from_float(m_grad_input_fp32_[i], (uint8_t*)input_grad + i * config_.hidden_size * ggml_type_size(config_.grad_type), config_.hidden_size, config_.grad_type);
    }, nullptr);
}

void SFT_MOE::backward(int layer_idx, int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, const void* grad_output, void* grad_input, Backend* backend) {
	std::cout << "[backward] layer_idx: " << layer_idx << std::endl;
	// for (uint64_t eid = 0; eid < (uint64_t)config_.expert_num; ++eid) {
    //     int rows = m_local_num_[eid];
    //     if (rows == 0) continue;

    //     const int H = config_.hidden_size;
    //     const int I = config_.intermediate_size;
    //     size_t rowsH = (size_t)rows * H;
    //     size_t rowsI = (size_t)rows * I;

	// 	dump_grad_bin("layer0_E_bef_many"+std::to_string(eid)+"_up_proj_t", (uint8_t*)up_proj_t_ + eid * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.grad_type), (size_t)config_.hidden_size * config_.intermediate_size, config_.grad_type);

	// 	dump_grad_bin("layer0_E_bef_many"+std::to_string(eid)+"_up_proj", (uint8_t*)up_proj_ + eid * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.up_type), (size_t)config_.hidden_size * config_.intermediate_size, config_.up_type);

	// // 	dump_grad_bin("layer0_E_End"+std::to_string(eid)+"_gate_proj_t", gate_proj_t_ + eid * config_.hidden_size * config_.intermediate_size, (size_t)config_.hidden_size * config_.intermediate_size, config_.grad_type);

	// // 	dump_grad_bin("layer0_E_End"+std::to_string(eid)+"_gate_proj", gate_proj_ + eid * config_.hidden_size * config_.intermediate_size, (size_t)config_.hidden_size * config_.intermediate_size, config_.gate_type);

	// // 	dump_grad_bin("layer0_E_End"+std::to_string(eid)+"_down_proj_t", down_proj_t_ + eid * config_.hidden_size * config_.intermediate_size, (size_t)config_.hidden_size * config_.intermediate_size, config_.grad_type);

	// // 	dump_grad_bin("layer0_E_End"+std::to_string(eid)+"_down_proj", down_proj_ + eid * config_.hidden_size * config_.intermediate_size, (size_t)config_.hidden_size * config_.intermediate_size, config_.down_type);
	// }

    int remaining_qlen = qlen;
    int processed_offset = 0;
    
    while (remaining_qlen > 0) {
        // config_.group_min_len = 10000000;
        if (remaining_qlen < config_.group_min_len) {
            for (int i = 0; i < remaining_qlen; i++) {
                backward_one(layer_idx, k,
                             expert_ids + (processed_offset + i) * k,
                             weights + (processed_offset + i) * k,
                             (uint8_t*)input + (processed_offset + i) * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type),
                             (uint8_t*)grad_output + (processed_offset + i) * config_.hidden_size * ggml_type_size(config_.grad_type),
                             (uint8_t*)grad_input + (processed_offset + i) * config_.hidden_size * ggml_type_size(config_.grad_type),
                             backend);
            }
            break;
        } else {
            int backward_len = std::min(config_.group_max_len, remaining_qlen);
            backward_many(layer_idx, backward_len, k, 
                         expert_ids + processed_offset * k, 
                         weights + processed_offset * k, 
                         (uint8_t*)input + processed_offset * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type),
                         (uint8_t*)grad_output + processed_offset * config_.hidden_size * ggml_type_size(config_.grad_type), 
                         (uint8_t*)grad_input + processed_offset * config_.hidden_size * ggml_type_size(config_.grad_type), 
                         backend);
            
            remaining_qlen -= backward_len;
            processed_offset += backward_len;
        }
    }
}