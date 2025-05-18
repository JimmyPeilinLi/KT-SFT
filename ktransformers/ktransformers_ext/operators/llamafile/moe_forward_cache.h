#pragma once
#include <vector>

struct MoEForwardCache {
    // 每个 token 按 expert 分块保存
    std::vector<std::vector<float>> gate_u;   // u = W_gate x
    std::vector<std::vector<float>> up_v;     // v = W_up   x
    // 若希望反向直接用 z = σ(u)⊙v，则再加一份
    // std::vector<std::vector<float>> z;        // 可选

    // helper：一次性按 k, inter_size 分配
    void init(int k, int inter_size, bool keep_z = false) {
        gate_u.assign(k, std::vector<float>(inter_size));
        up_v  .assign(k, std::vector<float>(inter_size));
        // if (keep_z) z.assign(k, std::vector<float>(inter_size));
    }
};