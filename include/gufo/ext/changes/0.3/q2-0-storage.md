# Q2_0 GGUF storage and row decoding

Add `core::GgmlType::kQ2_0` (GGML type 42). Quantized storage helpers and
`quant::DequantizeRow` accept its 64-element, 18-byte blocks. This adds
storage/CPU decoding support; model loaders and GPU kernels must separately
qualify their compute paths before accepting Q2_0 weights.
