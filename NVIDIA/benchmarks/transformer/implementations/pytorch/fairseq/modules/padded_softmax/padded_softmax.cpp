#include <torch/torch.h>
#include <cuda_fp16.h>

bool fast_softmax(__half *dst, __half *src, int softmax_elements, int softmax_elements_stride, int batch_count);
bool masked_fast_softmax(__half *dst, __half *src, uint8_t *pad_mask, int softmax_elements, int softmax_elements_stride, int batch_count, int pad_batch_stride);
bool softmax_backward(__half *grad_input, __half *grad, __half *output, int softmax_elements, int softmax_elements_stride, int batch_count);
bool masked_softmax_backward(__half *grad_input, __half *grad, __half *output, uint8_t *pad_mask, int softmax_elements, int softmax_elements_stride, int batch_count, int pad_batch_stride);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor fast_softmax_forward(
    at::Tensor result,
    at::Tensor input) {
  //CHECK_INPUT(in_result);
  //CHECK_INPUT(batch1);
  //CHECK_INPUT(batch2);

  AT_ASSERTM(result.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(input.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(input.size(0) == result.size(0), "equal number of batches expected");

  AT_ASSERTM(input.size(1) == result.size(1), "wrong matrix size");
  AT_ASSERTM(input.size(2) == result.size(2), "wrong matrix size");

  AT_ASSERTM(input.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(result.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");

  fast_softmax(reinterpret_cast<__half*>(result.data_ptr()), reinterpret_cast<__half*>(input.data_ptr()), result.size(2), result.stride(1), result.size(0) * result.size(1));

  return result;
}

at::Tensor fast_softmax_backward(
    at::Tensor grad_input,
    at::Tensor grad,
    at::Tensor output) {
  //CHECK_INPUT(in_result);
  //CHECK_INPUT(batch1);
  //CHECK_INPUT(batch2);

  AT_ASSERTM(grad_input.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(grad.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(output.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(grad_input.size(0) == grad.size(0), "equal number of batches expected");
  AT_ASSERTM(grad_input.size(0) == output.size(0), "equal number of batches expected");

  AT_ASSERTM(grad_input.size(1) == grad.size(1), "wrong matrix size");
  AT_ASSERTM(grad_input.size(2) == grad.size(2), "wrong matrix size");

  AT_ASSERTM(grad_input.size(1) == output.size(1), "wrong matrix size");
  AT_ASSERTM(grad_input.size(2) == output.size(2), "wrong matrix size");

  AT_ASSERTM(grad_input.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(grad.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");

  softmax_backward(reinterpret_cast<__half*>(grad_input.data_ptr()), reinterpret_cast<__half*>(grad.data_ptr()), reinterpret_cast<__half*>(output.data_ptr()), grad.size(2), grad.stride(1), grad.size(0) * grad.size(1));

  return grad_input;
}

at::Tensor masked_fast_softmax_forward(
    at::Tensor result,
    at::Tensor input,
    at::Tensor mask) {
  //CHECK_INPUT(in_result);
  //CHECK_INPUT(batch1);
  //CHECK_INPUT(batch2);

  AT_ASSERTM(result.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(input.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(input.size(0) == result.size(0), "equal number of batches expected");

  AT_ASSERTM(input.size(1) == result.size(1), "wrong matrix size");
  AT_ASSERTM(input.size(2) == result.size(2), "wrong matrix size");

  AT_ASSERTM(input.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(result.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(mask.type().scalarType()   == at::ScalarType::Byte, "Only BYTE is supported");

  masked_fast_softmax(reinterpret_cast<__half*>(result.data_ptr()), reinterpret_cast<__half*>(input.data_ptr()), reinterpret_cast<uint8_t*>(mask.data_ptr()), result.size(2), result.stride(1), result.size(0) * result.size(1),
      (result.size(0) * result.size(1)) / mask.size(0));

  return result;
}

at::Tensor masked_fast_softmax_backward(
    at::Tensor grad_input,
    at::Tensor grad,
    at::Tensor output,
    at::Tensor mask) {
  //CHECK_INPUT(in_result);
  //CHECK_INPUT(batch1);
  //CHECK_INPUT(batch2);

  AT_ASSERTM(grad_input.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(grad.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(output.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(grad_input.size(0) == grad.size(0), "equal number of batches expected");
  AT_ASSERTM(grad_input.size(0) == output.size(0), "equal number of batches expected");

  AT_ASSERTM(grad_input.size(1) == grad.size(1), "wrong matrix size");
  AT_ASSERTM(grad_input.size(2) == grad.size(2), "wrong matrix size");

  AT_ASSERTM(grad_input.size(1) == output.size(1), "wrong matrix size");
  AT_ASSERTM(grad_input.size(2) == output.size(2), "wrong matrix size");

  AT_ASSERTM(grad_input.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(grad.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");

  masked_softmax_backward(reinterpret_cast<__half*>(grad_input.data_ptr()), reinterpret_cast<__half*>(grad.data_ptr()), reinterpret_cast<__half*>(output.data_ptr()), reinterpret_cast<uint8_t*>(mask.data_ptr()), grad.size(2), grad.stride(1), grad.size(0) * grad.size(1), (grad.size(0) * grad.size(1)) / mask.size(0));

  return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("fast_forward", &fast_softmax_forward, "optimized softmax for small last dim forward.");
        m.def("masked_fast_forward", &masked_fast_softmax_forward, "optimized softmax for small last dim forward.");
        m.def("fast_backward", &fast_softmax_backward, "optimized softmax for small last dim backward.");
        m.def("masked_fast_backward", &masked_fast_softmax_backward, "optimized softmax for small last dim backward.");
}
