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
 * Copyright (c) 2015 by Contributors
 * \file add_relu-inl.h
 * \brief add relu family operator
 * \author Clement Fuji Tsang
*/
#ifndef MXNET_OPERATOR_ADD_RELU_INL_H_
#define MXNET_OPERATOR_ADD_RELU_INL_H_


#include <vector>
#include <algorithm>
#include "./tensor/elemwise_binary_op.h"
#include "./tensor/elemwise_binary_op-inl.h"
namespace mxnet {
namespace op {

struct AddReluKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* lhs,
    const DType* rhs, const OpReqType req) {
    DType tmp = lhs[i] + rhs[i];
    KERNEL_ASSIGN(out_data[i], req, tmp < DType(0) ? DType(0) : tmp);
  }
};
#if MSHADOW_CUDA_HALF2
template<>
MSHADOW_XINLINE void AddReluKernel::Map<mshadow::half::half2_t>(int i,
  mshadow::half::half2_t* out_data, const mshadow::half::half2_t* lhs,
  const mshadow::half::half2_t* rhs, const OpReqType req) {
  float low = __low2float(lhs[i].half2_) + __low2float(rhs[i].half2_);
  float high = __high2float(lhs[i].half2_) + __high2float(rhs[i].half2_);
  KERNEL_ASSIGN(out_data[i], req,
    mshadow::half::half2_t(__floats2half2_rn(
    low < 0.f ? 0.f : low, high < 0.f ? 0.f : high)));
}
#else
template<>
MSHADOW_XINLINE void AddReluKernel::Map<mshadow::half::half2_t>(int i,
  mshadow::half::half2_t* out_data, const mshadow::half::half2_t* lhs,
  const mshadow::half::half2_t* rhs, const OpReqType req) {
  float low = static_cast<float>(lhs[i].half_t2[0]) + static_cast<float>(rhs[i].half_t2[0]);
  float high = static_cast<float>(lhs[i].half_t2[1]) + static_cast<float>(rhs[i].half_t2[1]);
  KERNEL_ASSIGN(out_data[i], req,
    mshadow::half::half2_t(mshadow::half::half_t(low < 0.f ? 0.f : low), \
                           mshadow::half::half_t(high < 0.f ? 0.f : high)));
}
#endif

template<typename xpu>
inline void AddReluCpu(const nnvm::NodeAttrs& attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Kernel<AddReluKernel, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
      inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), req[0]);
  });
}


template<typename xpu>
inline void AddReluGpu(const nnvm::NodeAttrs& attrs,
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
      Kernel<AddReluKernel, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
        inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), req[0]);
    }
    break;
  case mshadow::kFloat64:
    {
      typedef double DType;
      Kernel<AddReluKernel, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
        inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), req[0]);
    }
    break;
  case mshadow::kFloat16:
    {
      typedef mshadow::half::half2_t DType;
      Kernel<AddReluKernel, xpu>::Launch(s, (inputs[0].Size() + 1) / 2, outputs[0].dptr<DType>(),
        inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), req[0]);
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

struct AddReluGradKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* input_grad,
    const DType* out_grad, const DType* out_data, const OpReqType req) {
    KERNEL_ASSIGN(input_grad[i], req,
      out_data[i] == DType(0.) ? DType(0.) : out_grad[i]);
  }
};
#if MSHADOW_CUDA_HALF2
template<>
MSHADOW_XINLINE void AddReluGradKernel::Map<mshadow::half::half2_t>(int i,
  mshadow::half::half2_t* input_grad, const mshadow::half::half2_t* out_grad,
  const mshadow::half::half2_t* out_data, const OpReqType req) {
  KERNEL_ASSIGN(input_grad[i], req,
    mshadow::half::half2_t(__floats2half2_rn(
      __low2float(out_data[i].half2_) == 0.f ?
      __half(0.) : __low2half(out_grad[i].half2_),
      __high2float(out_data[i].half2_) == 0.f ?
      __half(0.) : __high2half(out_grad[i].half2_))));
}
#else
template<>
MSHADOW_XINLINE void AddReluGradKernel::Map<mshadow::half::half2_t>(int i,
  mshadow::half::half2_t* input_grad, const mshadow::half::half2_t* out_grad,
  const mshadow::half::half2_t* out_data, const OpReqType req) {
  KERNEL_ASSIGN(input_grad[i], req, mshadow::half::half2_t(
    out_data[i].half_t2[0] == mshadow::half::half_t(0.) ?
    mshadow::half::half_t(0.) : out_grad[i].half_t2[0],
    out_data[i].half_t2[1] == mshadow::half::half_t(0.) ?
    mshadow::half::half_t(0.) : out_grad[i].half_t2[1]));
}
#endif


template<typename xpu>
inline void AddReluGradCpu(const nnvm::NodeAttrs& attrs,
                        const OpContext &ctx,
                        const std::vector<TBlob> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Kernel<AddReluGradKernel, xpu>::Launch(s, inputs[0].Size(),
      outputs[0].dptr<DType>(), inputs[0].dptr<DType>(),
      inputs[1].dptr<DType>(), req[0]);
  });
}

template<typename xpu>
inline void AddReluGradGpu(const nnvm::NodeAttrs& attrs,
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
      Kernel<AddReluGradKernel, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
        inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), req[0]);
    }
    break;
  case mshadow::kFloat64:
    {
      typedef double DType;
      Kernel<AddReluGradKernel, xpu>::Launch(s, inputs[0].Size(), outputs[0].dptr<DType>(),
        inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), req[0]);
    }
    break;
  case mshadow::kFloat16:
    {
      typedef mshadow::half::half2_t DType;
      Kernel<AddReluGradKernel, xpu>::Launch(s, (inputs[0].Size() + 1) / 2,
        outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), req[0]);
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

struct DoubleElemwiseGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    heads.emplace_back(nnvm::NodeEntry{n, 0, 0});
    if (CheckGradAllZero(ograds)) return MakeZeroGradNodes(n, ograds);
    auto p = MakeNode(op_name, n->attrs.name + "_backward",
                      nullptr, &(n->attrs.dict), &n);
    p->inputs.emplace_back(ograds[0]);
    p->inputs.emplace_back(heads[0]);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p, 0, 0});
    return ret;
  }
};





}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ADD_RELU_INL_H_

