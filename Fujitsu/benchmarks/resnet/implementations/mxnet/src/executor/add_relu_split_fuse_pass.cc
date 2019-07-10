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
 * \file add_relu_fuse_pass.cc
 * \brief detect and fuse whenever fused add_relu + split is possible on backward graph
 * \author Clement Fuji Tsang
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

#include "./exec_pass.h"

namespace mxnet {
namespace exec {

Graph FuseAddReluSplit(Graph&& g) {
  static const Op* backward_add_relu_op = nnvm::Op::Get("_backward_add_relu");
  static const Op* add_n_op = nnvm::Op::Get("add_n");
  static const Op* backard_add_relu_split_op = nnvm::Op::Get("_backward_add_relu_split");

  DFSVisit(g.outputs, [](const nnvm::NodePtr &node) {
    if (node->op() == backward_add_relu_op &&
        node->inputs[0].node->op() == add_n_op &&
        node->inputs[0].node->inputs.size() == 2 &&
        node->inputs[0].node.unique()) {
      // We must create a 2nd temporary shared_ptr to the input node so that
      // the node is not deleted in the midst of copying out some of its data members.
      auto held_input_reference = node->inputs[0].node;
      auto add_n_output = node->inputs[1];
      // Convert the _backward_add_relu node to _backward_add_relu_split, discard the add_n
      node->attrs.name = node->attrs.name + "_split";
      node->attrs.op = backard_add_relu_split_op;
      // Copy the add_n inputs to the _backward_add_relu_split inputs, discarding one of the
      // two references to the add_n in the process.
      node->inputs = node->inputs[0].node->inputs;
      // Retain the original 2nd input to the _backward_add_relu (the forward add_relu output)
      node->inputs.push_back(add_n_output);
      // Discard last reference to add_n, causing its destruction.
      held_input_reference.reset();
    }
  });

  return g;
}

}  // namespace exec
}  // namespace mxnet

