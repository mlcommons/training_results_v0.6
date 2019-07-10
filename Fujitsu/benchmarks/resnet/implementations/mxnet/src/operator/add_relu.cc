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
 * \file add_relu.cc
 * \brief
 * \author Clement Fuji Tsang
*/

#include "./add_relu-inl.h"

#include <nnvm/op_attr_types.h>
namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_BINARY(add_relu)
.MXNET_DESCRIBE("desc")
.set_attr<FInferStorageType>("FInferStorageType",
  ElemwiseStorageType<2, 1, false, false, false>)
.set_attr<FCompute>("FCompute<cpu>", AddReluCpu<cpu>)
.set_attr<nnvm::FGradient>("FGradient", DoubleElemwiseGrad{"_backward_add_relu"});

NNVM_REGISTER_OP(_backward_add_relu)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 0},
                                                                           {0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", AddReluGradCpu<cpu>)
.set_attr<FInferStorageType>("FInferStorageType",
  ElemwiseStorageType<2, 1, false, false, false>);

}  // namespace op
}  // namespace mxnet
