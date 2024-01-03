# deployOnnxGraphOpt

About project deployment onnx graph optimization.

~~Note: Some functions of our program can only be guaranteed to run correctly when onnx opset_version=11, such as opset!=11, please configure the conversion opset, it may be possible to avoid errors but not necessarily.~~

~~**Supplement: MobileViT-v2 already supports opset_version >= 11.**Other networks or modules do not currently support higher versions and will be upgraded in the future.~~


*Update：

1. Supports onnx model version network conversion for opset_version <= 18.**Fixed opset_version==11 is no longer required.
2. Built-in onnx model upgrade, configure the target opset_version through args (-v).
3. Added support for MobileViT-v1 and MobileViT-v2.
