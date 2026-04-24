#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void binary_sta_fused_forward_fp32_kernel(
    const int64_t* __restrict__ packed,
    const float* __restrict__ values,
    float* __restrict__ out,
    int B,
    int L,
    int W,
    int D,
    int num_planes,
    bool causal,
    bool use_softmax,
    float temperature) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  const int q = blockIdx.y;
  const int b = blockIdx.z;

  if (d >= D || q >= L || b >= B) {
    return;
  }

  const int packed_base = (b * L) * W;
  const int value_base = (b * L) * D;

  float max_logit = -1e30f;
  if (use_softmax) {
    for (int k = 0; k < L; ++k) {
      if (causal && k > q) {
        continue;
      }
      int mismatch = 0;
      const int q_off = packed_base + q * W;
      const int k_off = packed_base + k * W;
      for (int w = 0; w < W; ++w) {
        uint32_t x = static_cast<uint32_t>(packed[q_off + w] ^ packed[k_off + w]);
        mismatch += __popc(x);
      }
      float sim = static_cast<float>(num_planes - mismatch) / static_cast<float>(num_planes);
      float logit = sim / temperature;
      if (logit > max_logit) {
        max_logit = logit;
      }
    }
  }

  float denom = 0.0f;
  if (use_softmax) {
    for (int k = 0; k < L; ++k) {
      if (causal && k > q) {
        continue;
      }
      int mismatch = 0;
      const int q_off = packed_base + q * W;
      const int k_off = packed_base + k * W;
      for (int w = 0; w < W; ++w) {
        uint32_t x = static_cast<uint32_t>(packed[q_off + w] ^ packed[k_off + w]);
        mismatch += __popc(x);
      }
      float sim = static_cast<float>(num_planes - mismatch) / static_cast<float>(num_planes);
      denom += __expf(sim / temperature - max_logit);
    }
    denom = fmaxf(denom, 1e-8f);
  } else {
    for (int k = 0; k < L; ++k) {
      if (causal && k > q) {
        continue;
      }
      int mismatch = 0;
      const int q_off = packed_base + q * W;
      const int k_off = packed_base + k * W;
      for (int w = 0; w < W; ++w) {
        uint32_t x = static_cast<uint32_t>(packed[q_off + w] ^ packed[k_off + w]);
        mismatch += __popc(x);
      }
      float sim = static_cast<float>(num_planes - mismatch) / static_cast<float>(num_planes);
      denom += sim;
    }
    denom = fmaxf(denom, 1e-8f);
  }

  float acc = 0.0f;
  for (int k = 0; k < L; ++k) {
    if (causal && k > q) {
      continue;
    }
    int mismatch = 0;
    const int q_off = packed_base + q * W;
    const int k_off = packed_base + k * W;
    for (int w = 0; w < W; ++w) {
      uint32_t x = static_cast<uint32_t>(packed[q_off + w] ^ packed[k_off + w]);
      mismatch += __popc(x);
    }
    float sim = static_cast<float>(num_planes - mismatch) / static_cast<float>(num_planes);
    float wgt = use_softmax ? (__expf(sim / temperature - max_logit) / denom) : (sim / denom);
    acc += wgt * values[value_base + k * D + d];
  }

  out[value_base + q * D + d] = acc;
}

__global__ void binary_sta_fused_forward_fp16_kernel(
    const int64_t* __restrict__ packed,
    const half* __restrict__ values,
    half* __restrict__ out,
    int B,
    int L,
    int W,
    int D,
    int num_planes,
    bool causal,
    bool use_softmax,
    float temperature) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  const int q = blockIdx.y;
  const int b = blockIdx.z;

  if (d >= D || q >= L || b >= B) {
    return;
  }

  const int packed_base = (b * L) * W;
  const int value_base = (b * L) * D;

  float max_logit = -1e30f;
  if (use_softmax) {
    for (int k = 0; k < L; ++k) {
      if (causal && k > q) {
        continue;
      }
      int mismatch = 0;
      const int q_off = packed_base + q * W;
      const int k_off = packed_base + k * W;
      for (int w = 0; w < W; ++w) {
        uint32_t x = static_cast<uint32_t>(packed[q_off + w] ^ packed[k_off + w]);
        mismatch += __popc(x);
      }
      float sim = static_cast<float>(num_planes - mismatch) / static_cast<float>(num_planes);
      float logit = sim / temperature;
      if (logit > max_logit) {
        max_logit = logit;
      }
    }
  }

  float denom = 0.0f;
  if (use_softmax) {
    for (int k = 0; k < L; ++k) {
      if (causal && k > q) {
        continue;
      }
      int mismatch = 0;
      const int q_off = packed_base + q * W;
      const int k_off = packed_base + k * W;
      for (int w = 0; w < W; ++w) {
        uint32_t x = static_cast<uint32_t>(packed[q_off + w] ^ packed[k_off + w]);
        mismatch += __popc(x);
      }
      float sim = static_cast<float>(num_planes - mismatch) / static_cast<float>(num_planes);
      denom += __expf(sim / temperature - max_logit);
    }
    denom = fmaxf(denom, 1e-8f);
  } else {
    for (int k = 0; k < L; ++k) {
      if (causal && k > q) {
        continue;
      }
      int mismatch = 0;
      const int q_off = packed_base + q * W;
      const int k_off = packed_base + k * W;
      for (int w = 0; w < W; ++w) {
        uint32_t x = static_cast<uint32_t>(packed[q_off + w] ^ packed[k_off + w]);
        mismatch += __popc(x);
      }
      float sim = static_cast<float>(num_planes - mismatch) / static_cast<float>(num_planes);
      denom += sim;
    }
    denom = fmaxf(denom, 1e-8f);
  }

  float acc = 0.0f;
  for (int k = 0; k < L; ++k) {
    if (causal && k > q) {
      continue;
    }
    int mismatch = 0;
    const int q_off = packed_base + q * W;
    const int k_off = packed_base + k * W;
    for (int w = 0; w < W; ++w) {
      uint32_t x = static_cast<uint32_t>(packed[q_off + w] ^ packed[k_off + w]);
      mismatch += __popc(x);
    }
    float sim = static_cast<float>(num_planes - mismatch) / static_cast<float>(num_planes);
    float wgt = use_softmax ? (__expf(sim / temperature - max_logit) / denom) : (sim / denom);
    acc += wgt * __half2float(values[value_base + k * D + d]);
  }

  out[value_base + q * D + d] = __float2half(acc);
}

}  // namespace

torch::Tensor binary_sta_fused_forward_cuda(
    torch::Tensor packed_codes,
    torch::Tensor values,
    int64_t num_planes,
    int64_t chunk_size,
    bool causal,
    bool use_softmax,
    double temperature) {
  (void)chunk_size;
  const auto B = static_cast<int>(packed_codes.size(0));
  const auto L = static_cast<int>(packed_codes.size(1));
  const auto W = static_cast<int>(packed_codes.size(2));
  const auto D = static_cast<int>(values.size(2));

  auto out = torch::zeros_like(values);
  const int threads = 256;
  const dim3 blocks((D + threads - 1) / threads, L, B);

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  if (values.scalar_type() == torch::kFloat) {
    binary_sta_fused_forward_fp32_kernel<<<blocks, threads, 0, stream>>>(
        packed_codes.data_ptr<int64_t>(),
        values.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        L,
        W,
        D,
        static_cast<int>(num_planes),
        causal,
        use_softmax,
        static_cast<float>(temperature));
  } else {
    binary_sta_fused_forward_fp16_kernel<<<blocks, threads, 0, stream>>>(
        packed_codes.data_ptr<int64_t>(),
        reinterpret_cast<half*>(values.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B,
        L,
        W,
        D,
        static_cast<int>(num_planes),
        causal,
        use_softmax,
        static_cast<float>(temperature));
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
