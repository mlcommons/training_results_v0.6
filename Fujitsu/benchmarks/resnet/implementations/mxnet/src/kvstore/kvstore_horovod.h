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

/**
 * Copyright (c) 2018 by Contributors
 * @file   kvstore_dist_sync_allreduce.h
 * @brief  distributed implementation based on Horovod
 */
#ifndef MXNET_KVSTORE_KVSTORE_HOROVOD_H_
#define MXNET_KVSTORE_KVSTORE_HOROVOD_H_

#include <mxnet/kvstore.h>
#include <unordered_map>
#include <bitset>
#include <vector>
#include <string>
#include <utility>
#include <functional>
#include <algorithm>
#include "./comm.h"
#include "./kvstore_utils.h"
#include "./kvstore_local.h"

#if MXNET_USE_HOROVOD
#include "horovod/common/operations.h"
#include "horovod/mxnet/mpi_ops.h"

namespace mxnet {
namespace kvstore {

using namespace horovod::MX;
using namespace horovod::common;

/**
 * \brief store data in local machine
 */
class KVStoreHorovod : public KVStoreLocal {
 public:
  explicit KVStoreHorovod(bool use_device_comm)
  : KVStoreLocal(use_device_comm) {
    horovod_init();
  }

  virtual ~KVStoreHorovod() {
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void Pull(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            int priority,
            bool ignore_sparse) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void PullRowSparse(const std::vector<int>& keys,
                     const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                     int priority = 0) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void Push(const std::vector<std::string>& str_keys,
            const std::vector<NDArray>& values,
            int priority) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void Pull(const std::vector<std::string>& str_keys,
            const std::vector<NDArray*>& values,
            int priority,
            bool ignore_sparse) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void PullRowSparse(const std::vector<std::string>& str_keys,
                     const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                     int priority = 0) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void SetGradientCompression(const std::vector<std::pair<std::string, std::string> >
                              & kwargs) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void PushPull(const std::vector<int> &keys,
                const std::vector<NDArray> &in_values,
                const std::vector<NDArray*> &out_values,
                int priority,
                int average) override {
    SetKeyType(kIntKey);
    PushPullImpl(keys, in_values, out_values, priority, average);
  }

  void PushPull(const std::vector<std::string> &str_keys,
                const std::vector<NDArray> &in_values,
                const std::vector<NDArray*> &out_values,
                int priority,
                int average) override {
    SetKeyType(kStringKey);
    std::vector<int> keys(str_keys.size());
    LookupKeys(str_keys, &keys);
    PushPullImpl(keys, in_values, out_values, priority, average);
  }

  void Broadcast(const std::vector<int> &keys,
                 const std::vector<NDArray*> &values,
                 int root_rank,
                 int priority) override {
    SetKeyType(kIntKey);
    BroadcastImpl(keys, values, root_rank, priority);
  }

  void Broadcast(const std::vector<std::string> &str_keys,
                 const std::vector<NDArray*> &values,
                 int root_rank,
                 int priority) override {
    SetKeyType(kStringKey);
    std::vector<int> keys(str_keys.size());
    LookupKeys(str_keys, &keys);
    BroadcastImpl(keys, values, root_rank, priority);
  }

  // TODO(carlyang) Barrier may turn out to be necessary. Omitting for now, since
  // synchronization is taken care of in Horovod and it has no suitable API
  /*void Barrier() override {
    int ret = MXBarrier();
    if (ret == -1) {
      LOG(FATAL) << "MXBarrier is not successful. ret: " << ret;
    }
  }*/

  int get_rank() const override {
    int ret;
    ret = horovod_rank();
    if (ret == -1) {
      LOG(FATAL) << "horovod_rank is not successful. ret: " << ret;
    }
    return ret;
  }

  int get_local_rank() const override {
    int ret;
    ret = horovod_local_rank();
    if (ret == -1) {
      LOG(FATAL) << "horovod_local_rank is not successful. ret: " << ret;
    }
    return ret;
  }

  int get_group_size() const override {
    int ret;
    ret = horovod_size();
    if (ret == -1) {
      LOG(FATAL) << "horovod_size is not successful. ret: " << ret;
    }
    return ret;
  }

  int get_local_size() const override {
    int ret;
    ret = horovod_local_size();
    if (ret == -1) {
      LOG(FATAL) << "horovod_local_size is not successful. ret: " << ret;
    }
    return ret;
  }

 private:
  void InitImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values) override {
    CheckUnique(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      comm_->Init(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
    }
  }

  void PushPullImpl(const std::vector<int> &keys,
                    const std::vector<NDArray> &in_values,
                    const std::vector<NDArray*> &out_values,
                    int priority,
                    int average) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_invals;
    std::vector<std::vector<NDArray*> > grouped_outvals;

    CHECK_EQ(in_values.size(), out_values.size());
    GroupKVPairsPush(keys, in_values, &uniq_keys, &grouped_invals, false);
    uniq_keys.clear();
    GroupKVPairsPull(keys, out_values, &uniq_keys, &grouped_outvals, true);

    // reduce over devices
    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      char* name_from_key = new char[kIntLength];
      snprintf(name_from_key, kIntLength-1, "%s", std::to_string(key).c_str());

      NDArray& input = grouped_invals[i][0];
      NDArray& output = *grouped_outvals[i][0];
      auto allreduce_async_fn = [input, output, name_from_key, average](
          RunContext rctx, Engine::CallbackOnComplete cb) mutable {
        horovod_mxnet_allreduce_async(&input, &output, average, name_from_key, cb);
      };
      if (input.var() != output.var()) {
      Engine::Get()->PushAsync(
        allreduce_async_fn,
        input.ctx(),
        {input.var()},
        {output.var()},
        FnProperty::kNormal,
        priority,
        "KVStoreHorovodAllreduce");
      } else {
      Engine::Get()->PushAsync(
        allreduce_async_fn,
        input.ctx(),
        {},
        {output.var()},
        FnProperty::kNormal,
        priority,
        "KVStoreHorovodAllreduce");
      }
    }
  }

  void BroadcastImpl(const std::vector<int> &keys,
                     const std::vector<NDArray*> &values,
                     int root_rank,
                     int priority) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals, true);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      char* name_from_key = new char[kIntLength];
      snprintf(name_from_key, kIntLength-1, "%s", std::to_string(key).c_str());

      NDArray& input  = *grouped_vals[i][0];
      NDArray& output = *grouped_vals[i][0];
      // Create a temporary NDArray for in-place broadcast.
      NDArray input_cpu = NDArray(input.shape(), Context::CPU(), false, input.dtype());
      CopyFromTo(input, &input_cpu);
      auto broadcast_async_fn = [input_cpu, root_rank, name_from_key](
          RunContext rctx, Engine::CallbackOnComplete cb) mutable {
        horovod_mxnet_broadcast_async(&input_cpu, root_rank, name_from_key, cb);
      };
      Engine::Get()->PushAsync(
        broadcast_async_fn,
        comm_->pinned_ctx(),
        {},
        {output.var()},
        FnProperty::kNormal,
        priority,
        "KVStoreHorovodBroadcast");

      // Copy results to output.
      CopyFromTo(input_cpu, &output);
    }
  }

 private:
  std::unordered_map<int, NDArray> comm_buf_;

  const int kIntLength = 12;
};
}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET USE ALLREDUCE DIST KVSTORE
#endif  // MXNET_KVSTORE_KVSTORE_HOROVOD_H_
