/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file add_relu_split-inl.h
 * \brief fused backward pass of add_relu + copy/split (beginning of residual module)
 * \author Clement Fuji Tsang
*/
#ifndef MXNET_OPERATOR_ADD_RELU_SPLIT_INL_H_
#define MXNET_OPERATOR_ADD_RELU_SPLIT_INL_H_

#include <vector>
#include <algorithm>
#include "./tensor/elemwise_binary_op.h"
#include "./tensor/elemwise_binary_op-inl.h"
namespace mxnet {
namespace op {

struct AddReluSplitGradKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* input_grad, const DType* lhs_out_grad,
                                  const DType* rhs_out_grad, const DType* out_data,
                                  const OpReqType req) {
    DType tmp = lhs_out_grad[i] + rhs_out_grad[i];
    KERNEL_ASSIGN(input_grad[i], req, out_data[i] == DType(0.) ? DType(0.) : tmp);
  }
};

#if MSHADOW_CUDA_HALF2
template<>
MSHADOW_XINLINE void AddReluSplitGradKernel::Map<mshadow::half::half2_t>(int i,
    mshadow::half::half2_t* input_grad, const mshadow::half::half2_t* lhs_out_grad,
    const mshadow::half::half2_t* rhs_out_grad, const mshadow::half::half2_t* out_data,
    const OpReqType req) {
  float low = __low2float(lhs_out_grad[i].half2_) + __low2float(rhs_out_grad[i].half2_);
  float high = __high2float(lhs_out_grad[i].half2_) + __high2float(rhs_out_grad[i].half2_);
  KERNEL_ASSIGN(input_grad[i], req,  mshadow::half::half2_t(__floats2half2_rn(
                __low2float(out_data[i].half2_) == 0.f ? 0.f : low,
                __high2float(out_data[i].half2_) == 0.f ? 0.f : high)));
}
#else
template<>
MSHADOW_XINLINE void AddReluSplitGradKernel::Map<mshadow::half::half2_t>(int i,
    mshadow::half::half2_t* input_grad, const mshadow::half::half2_t* lhs_out_grad,
    const mshadow::half::half2_t* rhs_out_grad, const mshadow::half::half2_t* out_data,
    const OpReqType req) {
  mshadow::half::half_t low = lhs_out_grad[i].half_t2[0] + rhs_out_grad[i].half_t2[0];
  mshadow::half::half_t high = lhs_out_grad[i].half_t2[1] + rhs_out_grad[i].half_t2[1];
  KERNEL_ASSIGN(input_grad[i], req, mshadow::half::half2_t(
                out_data[i].half_t2[0] == mshadow::half::half_t(0.) ?
                mshadow::half::half_t(0.) : low,
                out_data[i].half_t2[1] == mshadow::half::half_t(0.) ?
                mshadow::half::half_t(0.) : high));
}
#endif


template<typename xpu>
inline void AddReluSplitGradCpu(const nnvm::NodeAttrs& attrs,
                                const OpContext &ctx,
                                const std::vector<TBlob> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Kernel<AddReluSplitGradKernel, xpu>::Launch(s, inputs[0].Size(),
      outputs[0].dptr<DType>(), inputs[0].dptr<DType>(),
      inputs[1].dptr<DType>(), inputs[2].dptr<DType>(), req[0]);
  });
}

template<typename xpu>
inline void AddReluSplitGradGpu(const nnvm::NodeAttrs& attrs,
                                const OpContext &ctx,
                                const std::vector<TBlob> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  switch (inputs[0].type_flag_) {
  case mshadow::kFloat32:
    {
      typedef float DType;
      Kernel<AddReluSplitGradKernel, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
        inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), inputs[2].dptr<DType>(), req[0]);
    }
    break;
  case mshadow::kFloat64:
    {
      typedef double DType;
      Kernel<AddReluSplitGradKernel, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
        inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), inputs[2].dptr<DType>(), req[0]);
    }
    break;
  case mshadow::kFloat16:
    {
      typedef mshadow::half::half2_t DType;
      Kernel<AddReluSplitGradKernel, xpu>::Launch(s, (inputs[0].Size() + 1) / 2,
        outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), inputs[1].dptr<DType>(),
        inputs[2].dptr<DType>(), req[0]);
    }
    break;
  case mshadow::kUint8:
    LOG(FATAL) << "This operation only support "
                  "floating point types not uint8";
    break;
  case mshadow::kInt8:
    LOG(FATAL) << "This operation only support "
                  "floating point types not int8";
    break;
  case mshadow::kInt32:
    LOG(FATAL) << "This operation only support "
                  "floating point types, not int32";
    break;
  case mshadow::kInt64:
    LOG(FATAL) << "This operation only support "
                  "floating point types, not int64";
    break;
  default:
    LOG(FATAL) << "Unknown type enum " << inputs[0].type_flag_;
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ADD_RELU_SPLIT_INL_H_


