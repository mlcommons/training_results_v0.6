#include <cuda_fp16.h>
#include <cstdint>
#include "softmax.h"

bool fast_softmax(__half *dst, __half *src, int softmax_elements, int softmax_elements_stride, int batch_count)
{
  return dispatch_softmax<__half, __half, float>(dst, (__half const*)src, softmax_elements, softmax_elements_stride, batch_count);
}
bool masked_fast_softmax(__half *dst, __half *src, uint8_t *pad_mask, int softmax_elements, int softmax_elements_stride, int batch_count, int pad_batch_stride)
{
  return dispatch_masked_softmax<__half, __half, float>(dst, (__half const*)src, (uint8_t const*)pad_mask, softmax_elements, softmax_elements_stride, batch_count, pad_batch_stride);
}
bool softmax_backward(__half *grad_input, __half *grad, __half *output, int softmax_elements, int softmax_elements_stride, int batch_count)
{
  return dispatch_softmax_backward<__half, __half, float>(grad_input, ( __half const*)grad, (__half const*)output, softmax_elements, softmax_elements_stride, batch_count);  
}

bool masked_softmax_backward(__half *grad_input, __half *grad, __half *output, uint8_t *pad_mask, int softmax_elements, int softmax_elements_stride, int batch_count, int pad_batch_stride)
{
  return dispatch_masked_softmax_backward<__half, __half, float>(grad_input, ( __half const*)grad, (__half const*)output, (uint8_t const*)pad_mask, softmax_elements, softmax_elements_stride, batch_count, pad_batch_stride);  
}

