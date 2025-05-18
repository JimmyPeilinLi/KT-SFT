/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-15 07:43:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "moe.h"
#include <iostream>
#include <cstdint>
#include <cstring>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

MOE::MOE(MOEConfig config) {
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
}

MOE::~MOE() {
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

void MOE::warm_up(Backend* backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
	/* ---------- 仅用于占位的 ForwardCache ---------- */
    MoEForwardCache dummy_cache; // 内容无用，只为满足接口
    for (int i = 0; i < config_.expert_num; i++) {
        uint64_t expert_ids = i;
        float weights = 0;
        forward_one(1, &expert_ids, &weights, input.data(), output.data(), backend, &dummy_cache);
    }
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}

void MOE::forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, MoEForwardCache* fwd_cache) {
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

	for (int e = 0; e < k; ++e) {
        // gate_output_: float[inter_size] per expert
        std::memcpy(fwd_cache->gate_u[e].data(),
                    s_gate_output_[e],
                    sizeof(float) * config_.intermediate_size);

        std::memcpy(fwd_cache->up_v[e].data(),
                    s_up_output_[e],
                    sizeof(float) * config_.intermediate_size);

        // 可选保存 z
        // std::memcpy(fwd_cache->z[e].data(),
        //             s_intermediate_fp32_[e],
        //             sizeof(float) * config_.intermediate_size);
    }
}

void MOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, MoEForwardCache* fwd_cache) {
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

	/* 把每个 token-expert 的行复制到各自 cache */
    backend->do_work_stealing_job(qlen, nullptr, [&](int token_idx) {
        auto& cache = fwd_cache[token_idx];
        // cache 已在上层 init(k, inter_size)
        for (int j = 0; j < k; ++j) {
            uint64_t  eid   = expert_ids[token_idx*k + j];
            int       row   = m_local_pos_[token_idx][j];
            size_t    ofs   = row * config_.intermediate_size;
            /* gate u */
            std::memcpy(cache.gate_u[j].data(),
                        m_local_gate_output_ptr_[eid] + ofs,
                        sizeof(float) * config_.intermediate_size);
            /* up v */
            std::memcpy(cache.up_v[j].data(),
                        m_local_up_output_ptr_[eid] + ofs,
                        sizeof(float) * config_.intermediate_size);
            /* 可选 z */
            // std::memcpy(cache.z[j].data(),
            //             m_local_intermediate_fp32_ptr_[eid] + ofs,
            //             sizeof(float) * config_.intermediate_size);
        }
    }, nullptr);
}

void MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, MoEForwardCache* fwd_cache) {
    if (qlen < config_.group_min_len) {
        for (int i = 0; i < qlen; i++) {
			fwd_cache[i].init(k, config_.intermediate_size);      // 预分配
            forward_one(k, expert_ids + i * k, weights + i * k, (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend, fwd_cache + i);
        }
        return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);
    for (int i = 0; i < forward_len; ++i)
        fwd_cache[i].init(k, config_.intermediate_size);
    forward_many(forward_len, k, expert_ids, weights, input, output, backend, fwd_cache);
    forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k, (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend, fwd_cache + forward_len);
}

void MOE::backward_one(int k, const uint64_t* expert_ids, const float* weights,
                       const void* input, const void* grad_output, void* grad_input, Backend* backend, const MoEForwardCache* fwd_cache) {
    // 准备梯度缓冲区
    float* grad_input_fp32 = new float[config_.hidden_size];
    float* grad_down_output = new float[config_.hidden_size];
    float** grad_intermediate = new float*[k];
    float** grad_gate_output = new float*[k];
    float** grad_up_output = new float*[k];

    for (int expert_idx = 0; expert_idx < k; expert_idx++) {
        grad_intermediate[expert_idx] = new float[config_.intermediate_size];
        grad_gate_output[expert_idx] = new float[config_.intermediate_size];
        grad_up_output[expert_idx] = new float[config_.intermediate_size];
    }

    // 初始化输入梯度为 0
    for (int i = 0; i < config_.hidden_size; i++) {
        grad_input_fp32[i] = 0.0f;
    }

    // 将 grad_output 转换为 float 类型
    to_float(grad_output, grad_down_output, config_.hidden_size, config_.hidden_type);

    // 对每个专家进行反向传播
    for (int expert_idx = 0; expert_idx < k; expert_idx++) {
        uint64_t expert_id = expert_ids[expert_idx];
        float expert_weight = weights[expert_idx];

        // 计算 down_output 的梯度
        float* expert_grad_down = new float[config_.hidden_size];
        for (int i = 0; i < config_.hidden_size; i++) {
            expert_grad_down[i] = grad_down_output[i] * expert_weight;
        }

        // 计算 intermediate 的梯度
        for (int i = 0; i < config_.intermediate_size; i++) {
            grad_intermediate[expert_idx][i] = 0.0f;
            for (int j = 0; j < config_.hidden_size; j++) {
                float down_proj_val = 0.0f;
                void* down_proj_ptr = (uint8_t*)down_proj_ +
                    (expert_id * config_.hidden_size * config_.intermediate_size +
                     j * config_.intermediate_size + i) *
                    ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
                to_float(down_proj_ptr, &down_proj_val, 1, config_.down_type);
                grad_intermediate[expert_idx][i] += down_proj_val * expert_grad_down[j];
            }
        }

        delete[] expert_grad_down;

		// 直接拿到 forward 的结果，目前不保存SiLU的计算中间结果（即z）
		const float* gate_output = fwd_cache->gate_u[expert_idx].data();
        const float* up_output   = fwd_cache->up_v [expert_idx].data();

        // SiLU 激活反向
        for (int i = 0; i < config_.intermediate_size; i++) {
            float x = gate_output[i];
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            float silu_x = x * sigmoid_x;
            float silu_derivative = sigmoid_x + x * sigmoid_x * (1.0f - sigmoid_x);

            grad_up_output[expert_idx][i] = grad_intermediate[expert_idx][i] * silu_x;
            grad_gate_output[expert_idx][i] = grad_intermediate[expert_idx][i] * up_output[i] * silu_derivative;
        }

        delete[] gate_output;
        delete[] up_output;

        // input 的梯度计算
        float* grad_input_from_gate = new float[config_.hidden_size];
        float* grad_input_from_up = new float[config_.hidden_size];

        for (int i = 0; i < config_.hidden_size; i++) {
            grad_input_from_gate[i] = 0.0f;
            grad_input_from_up[i] = 0.0f;
            for (int j = 0; j < config_.intermediate_size; j++) {
                float gate_proj_val = 0.0f;
                void* gate_proj_ptr = (uint8_t*)gate_proj_ +
                    (expert_id * config_.intermediate_size * config_.hidden_size +
                     j * config_.hidden_size + i) *
                    ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
                to_float(gate_proj_ptr, &gate_proj_val, 1, config_.gate_type);
                grad_input_from_gate[i] += gate_proj_val * grad_gate_output[expert_idx][j];

                float up_proj_val = 0.0f;
                void* up_proj_ptr = (uint8_t*)up_proj_ +
                    (expert_id * config_.intermediate_size * config_.hidden_size +
                     j * config_.hidden_size + i) *
                    ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
                to_float(up_proj_ptr, &up_proj_val, 1, config_.up_type);
                grad_input_from_up[i] += up_proj_val * grad_up_output[expert_idx][j];
            }

            grad_input_fp32[i] += (grad_input_from_gate[i] + grad_input_from_up[i]);
        }

        delete[] grad_input_from_gate;
        delete[] grad_input_from_up;
    }

    from_float(grad_input_fp32, grad_input, config_.hidden_size, config_.hidden_type);

    delete[] grad_input_fp32;
    delete[] grad_down_output;

    for (int expert_idx = 0; expert_idx < k; expert_idx++) {
        delete[] grad_intermediate[expert_idx];
        delete[] grad_gate_output[expert_idx];
        delete[] grad_up_output[expert_idx];
    }

    delete[] grad_intermediate;
    delete[] grad_gate_output;
    delete[] grad_up_output;
}

void MOE::backward(int qlen, int k, const uint64_t* expert_ids, const float* weights,
                   const void* input, const void* grad_output, void* grad_input, Backend* backend, const MoEForwardCache* fwd_cache) {
	std::memset(grad_input, 0, qlen * (config_.hidden_size * ggml_type_size(config_.hidden_type) /
    ggml_blck_size(config_.hidden_type)));

    for (int i = 0; i < qlen * config_.hidden_size * ggml_type_size(config_.hidden_type) /
         ggml_blck_size(config_.hidden_type); i++) {
        ((uint8_t*)grad_input)[i] = 0;
    }

    for (int i = 0; i < qlen; i++) {
        backward_one(k,
                     expert_ids + i * k,
                     weights + i * k,
                     (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type),
                     (uint8_t*)grad_output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type),
                     (uint8_t*)grad_input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type),
                     backend,
					 fwd_cache + i);
    }
}

// TODO: many拓展之后的调用方式
// void MOE::backward(int qlen, int k,
//                    const uint64_t* expert_ids,
//                    const float*     weights,
//                    const void*      input,
//                    const void*      grad_output,
//                    void*            grad_input,
//                    Backend*         backend,
//                    MoEForwardCache* fwd_cache)
// {
//     if (qlen < config_.group_min_len) {
//         for (int i = 0; i < qlen; ++i) {
//             backward_one(k,
//                          expert_ids + i * k,
//                          weights    + i * k,
//                          (const uint8_t*)input        + i * stride_in_bytes,
//                          (const uint8_t*)grad_output  + i * stride_in_bytes,
//                          (uint8_t*)grad_input         + i * stride_in_bytes,
//                          backend,
//                          fwd_cache + i);                // ⭐ 调用 backward_one
//         }
//         return;
//     }

//     int backward_len = std::min(config_.group_max_len, qlen);
//     backward_many(backward_len, k,                        // ⭐ 调用 backward_many
//                   expert_ids, weights,
//                   input, grad_output, grad_input,
//                   backend,
//                   fwd_cache);

//     backward(qlen - backward_len, k,                      // ⭐ 递归调用自己
//              expert_ids + backward_len * k,
//              weights    + backward_len * k,
//              (const uint8_t*)input        + backward_len * stride_in_bytes,
//              (const uint8_t*)grad_output  + backward_len * stride_in_bytes,
//              (uint8_t*)grad_input         + backward_len * stride_in_bytes,
//              backend,
//              fwd_cache + backward_len);
// }



// FIXME: expert backward second way to implement for C++
// 第二种可能的说法：
// void MOE::backward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* grad_output, Backend* backend) {
//     if (qlen < config_.group_min_len) {
//         for (int i = 0; i < qlen; i++) {
//             backward_one(k, expert_ids + i * k, weights + i * k, (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type),
//                          (uint8_t*)grad_output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
//         }
//         return;
//     }
//     int backward_len = std::min(config_.group_max_len, qlen);
//     backward_many(backward_len, k, expert_ids, weights, input, grad_output, backend);
//     backward(qlen - backward_len, k, expert_ids + backward_len * k, weights + backward_len * k, (uint8_t*)input + backward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type),
//              (uint8_t*)grad_output + backward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
// }

// void MOE::backward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* grad_output, Backend* backend) {
//     // Step 1: Compute the gradient of the output
//     const void* gate_input_ptr;
//     const void* up_input_ptr;

//     if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
//         gate_input_ptr = up_input_ptr = input;
//     } else {
//         to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
//         if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
//             from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
//             gate_input_ptr = up_input_ptr = s_gate_input_;
//         } else {
//             if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
//                 from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
//                 gate_input_ptr = s_gate_input_;
//             } else {
//                 gate_input_ptr = input;
//             }
//             if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
//                 from_float(s_input_fp32_, s_up_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
//                 up_input_ptr = s_up_input_;
//             } else {
//                 up_input_ptr = input;
//             }
//         }
//     }

//     // Step 2: Compute gradients for the gate and up projections
//     int nth = config_.intermediate_size / config_.stride;
//     backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
//         int expert_idx = task_id / nth;
//         uint64_t expert_id = expert_ids[expert_idx];
//         int ith = task_id % nth;

//         // Gradient for gate projection (backprop to weights and gate_input)
//         #ifdef USE_NUMA
//         void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
//         #else
//         void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
//         #endif

//         // Perform sgemm (backward pass) for gate projection
//         float* grad_gate_output_ptr = s_grad_gate_output_[expert_idx] + ith * config_.stride;
//         llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), grad_output, config_.hidden_size / ggml_blck_size(config_.gate_type), grad_gate_output_ptr, config_.stride, 1, 0, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

//         // Gradient for up projection (backprop to weights and up_input)
//         #ifdef USE_NUMA
//         void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
//         #else
//         void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
//         #endif

//         // Perform sgemm (backward pass) for up projection
//         float* grad_up_output_ptr = s_grad_up_output_[expert_idx] + ith * config_.stride;
//         llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), grad_gate_output_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), grad_up_output_ptr, config_.stride, 1, 0, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
//     }, nullptr);

//     // Step 3: Compute the gradient for the down projection and propagate upstream
//     // Use the weights and backprop the gradient to down projection
//     // Repeat similar steps as for gate and up projections...
// }
