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
 * \file add_relu_split.cc
 * \brief fused backward pass of add_relu + copy/split (beginning of residual module)
 * \author Clement Fuji Tsang
*/

#include "./add_relu_split-inl.h"

#include <nnvm/op_attr_types.h>
namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_backward_add_relu_split)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs", "out_grad"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 0},
                                                                           {0, 1},
                                                                           {0, 2}};
                                })
.add_argument("lhs", "NDArray-or-Symbol", "first input")
.add_argument("rhs", "NDArray-or-Symbol", "second input")
.add_argument("out_grad", "NDArray-or-Symbol", "")
.set_attr<FCompute>("FCompute<cpu>", AddReluSplitGradCpu<cpu>)
.set_attr<FInferStorageType>("FInferStorageType",
  ElemwiseStorageType<3, 1, false, false, false>);

}  // namespace op
}  // namespace mxnet

