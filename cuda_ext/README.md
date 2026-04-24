# Binary STA CUDA Extension

This folder contains a custom C++/CUDA extension for fused Binary STA forward:

- XOR + `__popc` similarity on packed uint32 words
- row-wise routing normalisation (`normalize` or `softmax`)
- value aggregation in one fused kernel path

The Python loader is in [binary_sta_cuda_ext.py](../binary_sta_cuda_ext.py).

## Build behavior

The extension is JIT-built at runtime via `torch.utils.cpp_extension.load` when backend is set to `cuda_ext`.

If build fails, the model automatically falls back to `packbits` backend.

Set `BINARY_STA_DISABLE_CUDA_EXT=1` to force-disable this path.
