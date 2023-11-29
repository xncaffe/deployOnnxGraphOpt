# deployOnnxGraphOpt

About project deployment onnx graph optimization.

Note: Some functions of our program can only be guaranteed to run correctly when onnx opset_version=11, such as opset!=11, please configure the conversion opset, it may be possible to avoid errors but not necessarily.

**Supplement: MobileViT-v2 already supports opset_version >= 11.**Other networks or modules do not currently support higher versions and will be upgraded in the future.
