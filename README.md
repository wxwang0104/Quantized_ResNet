# Quantized_ResNet
This repo is an implementation of quantized CNN for both weights (1-bit compression) and feature maps (2-bit compression) using Pytorch framework. 

Deep CNNs for classification and object detection take large stoarge space and computational cost. By quantizing the weight kernel and feature maps to low-bit representation, the storage space can be decreased to a large degree. At the same time bitwise operation can be much faster than float operation. Both problems can be solved in deep CNN. 

# Weight quantization

The original weights (32-bit FloatTensor) are quantized to 1-bit representation. Moreover, by multiplying a scaling factor of full precision with kernels in each layer, the solution space involve from point space to line space. 

<img src="weight.png" width="500">


