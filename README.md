# Insightface-go

Go implementation of Arcface inference.

Based on [Arcface-go](https://github.com/jack139/arcface-go),replace the onnxruntime-go&gocv repo.

## Prerequisites

- The onnx-format models used in the code is [&#34;**buffalo_l**&#34;](https://insightface.cn-sh2.ufileos.com/models/buffalo_l.zip) from [insightface](https://github.com/deepinsight/insightface/tree/master/model_zoo).
- [ONNX Runtime-GO](https://github.com/yalue/onnxruntime_go) Visit this repo,learn how to deploy onnxruntime with your own requirement (like Windows/Linux/Macos|Cuda/Openvino/Rocm/Coreml...).In my repo,it has been provided the dependencies needed for DirectML acceleration on the Windows platform.If you need Hardware acceleration in your ouw platform.You must prepare lib files and edit the file  . /arcface/arcface.go  in line 62.
- [GOCV](https://gocv.io/) is required, because some codes borrowed from gocv to implement EstimateAffinePartial2DWithParams().

## Run example

The example is too simple, detect faces in the input image and retrieve features of the first face.

```
go run example.go
```

- path to "**buffalo_l**" should be corrected in ``example.go`` .
