#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> fused_relu_dropout_cuda(
    const at::Tensor& input, 
    double prob);

at::Tensor fused_relu_dropout_backward_cuda(
    const at::Tensor& grad, 
    const at::Tensor& mask, 
    double scale);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> fused_relu_dropout(
    const at::Tensor& input, 
    double prob) {
  CHECK_CUDA(input);
  return fused_relu_dropout_cuda(input, prob);
}

at::Tensor fused_relu_dropout_backward(
    const at::Tensor& grad, 
    const at::Tensor& mask, 
    double scale) {
  CHECK_CUDA(grad);
  CHECK_CUDA(mask);
  return fused_relu_dropout_backward_cuda(grad, mask, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_relu_dropout, "Fused Relu and Dropout forward (CUDA)");
  m.def("backward", &fused_relu_dropout_backward, "Fused Relu and Dropout backward (CUDA)");
}
