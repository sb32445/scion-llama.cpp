#pragma once

#include "common.cuh"

#define GGML_CUDA_LORA_AB_BLOCKS_PER_SM 1

bool ggml_cuda_lora_ab_supported(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * x, int cc, int n_sm);

// dst = b * q8(a * q8(x)), PQ2_0 weights, one token
void ggml_cuda_lora_ab(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * x, ggml_tensor * dst);
