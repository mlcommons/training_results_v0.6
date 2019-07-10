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
 * Copyright (c) 2016 by Contributors
 * \file add_relu_fuse_pass.cc
 * \brief detect whether fused add + relu is possible
 * \author Clement Fuji Tsang
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

#include "./exec_pass.h"

namespace mxnet {
namespace exec {

Graph FuseAddRelu(Graph&& g) {
  static const Op* add_op = Op::Get("elemwise_add");
  static const Op* activation_op = Op::Get("Activation");
  static const Op* add_relu_op = nnvm::Op::Get("add_relu");

  DFSVisit(g.outputs, [](const nnvm::NodePtr &node) {
    if (node->op() == activation_op &&
        node->attrs.dict.at("act_type") == "relu" &&
        node->inputs[0].node->op() == add_op &&
        node->inputs[0].node.unique()) {
      // We must create a 2nd temporary shared_ptr to the input node so that
      // the node is not deleted in the midst of copying out some of its data members.
      auto held_input_reference = node->inputs[0].node;
      // Convert the Activation node to add_relu, discard the elemwise_add
      node->attrs.name = "add_" + node->attrs.name;
      node->attrs.op = add_relu_op;
      // Copy the elemwise_add inputs to the add_relu inputs, discarding one of the
      // two references to the elemwise_add in the process.
      node->inputs = node->inputs[0].node->inputs;
      // Discard last reference to elemwise_add, causing its destruction.
      held_input_reference.reset();
    }
  });

  return g;
}

}  // namespace exec
}  // namespace mxnet
