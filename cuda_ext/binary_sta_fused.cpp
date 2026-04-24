#include <torch/extension.h>

torch::Tensor binary_sta_fused_forward_cuda(
    torch::Tensor packed_codes,
    torch::Tensor values,
    int64_t num_planes,
    int64_t chunk_size,
    bool causal,
    bool use_softmax,
    double temperature);

torch::Tensor binary_sta_fused_forward(
    torch::Tensor packed_codes,
    torch::Tensor values,
    int64_t num_planes,
    int64_t chunk_size,
    bool causal,
    bool use_softmax,
    double temperature) {
  TORCH_CHECK(packed_codes.is_cuda(), "packed_codes must be CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA tensor");
  TORCH_CHECK(packed_codes.dim() == 3, "packed_codes must be [B, L, W]");
  TORCH_CHECK(values.dim() == 3, "values must be [B, L, D]");
  TORCH_CHECK(packed_codes.scalar_type() == torch::kInt64, "packed_codes must be int64");
  TORCH_CHECK(values.scalar_type() == torch::kFloat || values.scalar_type() == torch::kHalf,
              "values must be float16/float32");
  return binary_sta_fused_forward_cuda(
      packed_codes,
      values,
      num_planes,
      chunk_size,
      causal,
      use_softmax,
      temperature);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binary_sta_fused_forward", &binary_sta_fused_forward, "Binary STA fused forward (CUDA)");
}
