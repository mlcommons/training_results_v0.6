// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
// Modifications copyright (C) 2018 NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <atomic>
#include <cassert>
#include <cstring>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <mxnet/base.h>

#if HAVE_CUDA
#include <cuda_runtime.h>
#include "batched_memcpy.h"
#endif

#if HAVE_NCCL
#include <nccl.h>
#endif

#if HAVE_DDL
#include <ddl.hpp>
#endif

#define OMPI_SKIP_MPICXX
#include "hashes.h"
#include "mpi.h"
#include "mpi_message.h"
#include "operations.h"
#include "timeline.h"
#include "../mxnet/adapter.h"

#define ALIGN_BYTES 128

/*
 * Allreduce, Allgather and Broadcast Ops.
 *
 * This module implements MPI ops for allgather, allreduce and broadcast, which
 * do optimized gathers, reductions and broadcasts and can take advantage of
 * hardware-optimized communication libraries through the MPI implementation.
 *
 * The primary logic of the allreduce, allgather and broadcast are in MPI and
 * NCCL implementations. The background thread which facilitates MPI operations
 * is run in BackgroundThreadLoop(). The provided ops are:
 *      – HorovodAllreduce:
 *          Perform an allreduce on a Tensor, returning the sum
 *          across all MPI processes in the global communicator.
 *      – HorovodAllgather:
 *          Perform an allgather on a Tensor, returning the concatenation of
 *          the tensor on the first dimension across all MPI processes in the
 *          global communicator.
 *      - HorovodBroadcast:
 *          Perform a broadcast on a Tensor, broadcasting Tensor
 *          value from root rank to all other ranks.
 *
 * Additionally, this library provides C APIs to initialize Horovod and query
 * rank, local rank and world size.  These are used in Python directly through
 * ctypes.
 */

namespace horovod {
namespace common {

namespace {


std::vector<int> conv_text2numbers
( std::string str, char del )
{
  int first = 0;
  int last = str.find_first_of(del);
  
  std::vector<int> result;
  int number;

  while( first < str.size() ){
    std::string subStr( str, first, last - first );
    number = std::atoi( subStr.c_str() );
    result.push_back( number );

    first = last + 1;
    last = str.find_first_of( del, first);
    if( last == std::string::npos ){
      last = str.size();
    }
  }
  return result;
}



// Table storing Tensors to be reduced, keyed by unique name.
// This table contains everything necessary to do the reduction.
struct TensorTableEntry {
  // Name of the tensor.
  std::string tensor_name;
  // Operation context.
  std::shared_ptr<OpContext> context;
  // Input tensor.
  std::shared_ptr<Tensor> tensor;
  // Pre-allocated output tensor.
  std::shared_ptr<Tensor> output;
  // Root rank for broadcast operation.
  int root_rank = 0;
  // Event indicating that data is ready.
  std::shared_ptr<ReadyEvent> ready_event;
  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
  int device = CPU_DEVICE_ID;
  // A callback to call with the status.
  StatusCallback callback;
};
using TensorTable = std::unordered_map<std::string, TensorTableEntry>;

// Table for storing Tensor metadata on rank zero. This is used for error
// checking, stall checking and size calculations, as well as determining
// when a reduction is ready to be done (when all nodes are ready to do it).
using MessageTable = std::unordered_map<
    std::string,
    std::tuple<std::vector<MPIRequest>, std::chrono::steady_clock::time_point>>;

// Structure containing pinned host pointers for use with batched d2d copy
// kernel
#define PACK_PTRS_CAPACITY 500
struct PackPtrs {
  bool allocated = false;
  void** pack_out = nullptr;
  void** pack_in = nullptr;
  size_t* pack_sizes = nullptr;
  void** unpack_out = nullptr;
  void** unpack_in = nullptr;
  size_t* unpack_sizes = nullptr;

  void free() {
#if HAVE_CUDA
    if (allocated) {
      cudaFreeHost(pack_out);
      cudaFreeHost(pack_in);
      cudaFreeHost(pack_sizes);
      cudaFreeHost(unpack_out);
      cudaFreeHost(unpack_in);
      cudaFreeHost(unpack_sizes);
      allocated = false;
    }
#endif
  }
};

// The global state required for the MPI ops.
//
// MPI is a library that stores a lot of global per-program state and often
// requires running on a single thread. As a result, we have to have a single
// background thread responsible for all MPI operations, and communicate with
// that background thread through global state.
struct HorovodGlobalState {
  // An atomic boolean which is set to true when background thread is started.
  // This ensures that only one background thread is spawned.
  std::atomic_flag initialize_flag = ATOMIC_FLAG_INIT;

  // A mutex that needs to be used whenever MPI operations are done.
  std::mutex mutex;

  // Tensors waiting to be allreduced or allgathered.
  TensorTable tensor_table;

  // Queue of MPI requests waiting to be sent to the coordinator node.
  std::queue<MPIRequest> message_queue;

  // Background thread running MPI communication.
  std::thread background_thread;

  // Whether the background thread should shutdown.
  bool shut_down = false;

  // Only exists on the coordinator node (rank zero). Maintains a count of
  // how many nodes are ready to allreduce every tensor (keyed by tensor
  // name) and time point when tensor started allreduce op.
  std::unique_ptr<MessageTable> message_table;
  std::unique_ptr<MessageTable> local_message_table;
  std::unique_ptr<MessageTable> fixed_message_table;

  // Time point when coordinator last checked for stalled tensors.
  std::chrono::steady_clock::time_point last_stall_check;

  // Timeline writer.
  Timeline timeline;

  // Threshold for Tensor Fusion.  All tensors that occupy memory beyond this
  // threshold will be fused.
  int64_t tensor_fusion_threshold = 128 * 1024 * 1024; // "64 * 1024 * 1024;"

  // Background thread cycle time in milliseconds.  Fractional numbers are
  // permitted.
  double cycle_time_ms = 5;

  // Time point when last cycle started.
  std::chrono::steady_clock::time_point last_cycle_start;

  // Memory buffers for Tensor Fusion.  They are keyed off device ID and
  // framework, and all are allocated tensor_fusion_threshold bytes if
  // initialized.
  std::unordered_map<std::tuple<int, Framework>,
                     std::shared_ptr<PersistentBuffer>>
      tensor_fusion_buffers;

  PackPtrs pack_ptrs;

  // Whether MPI_Init has been completed on the background thread.
  bool initialization_done = false;

  // The MPI rank, local rank, size, local size and flag indicating whether MPI
  // multi-threading is supported.
  int rank = 0;
  int local_rank = 0;
  int cross_rank = 0;
  int size = 1;
  int local_size = 1;
  int cross_size = 1;
  bool mpi_threads_supported = false;
  bool is_homogeneous = false;

  // COMM_WORLD ranks of processes running on this node.
  std::vector<int> local_comm_ranks;

  // MPI custom data type for float16.
  MPI_Datatype mpi_float16_t;

  // Private MPI communicator for Horovod to ensure no collisions with other
  // threads using MPI.
  MPI_Comm mpi_comm;

  // Node-local communicator.
  MPI_Comm local_comm;

  // Cross-node communicator for hierarchical allreduce.
  MPI_Comm cross_comm;

  // Do hierarchical allreduce with MPI + NCCL.
  bool hierarchical_allreduce = false;

  // Use two stage control plane
  bool two_stage_loop = false;

  // Sets mode for allreduce (0: single global allreduce, 1: hierarchical on GPU)
  int allreduce_mode = 0;

  // Fixed number of tensors to allreduce in a step
  int fixed_payload = 0;

  // add new environment variables
  bool multiple_transfer_mode = false;
  int fixed_transfersize = 0;
  int num_transferred_layer = 0;
  bool quadratic_transfer_mode= false;
  bool cubic_transfer_mode= false;
  int layer_offset = 0;

  // broadcast never comes
  bool initial_Bcast_done = true;

  std::queue<MPIRequest> fp32_message_queue;
  int fp32_layer_count = 0;

  bool triple_streams4allreduce = false;
  int64_t transfer_buffer_offset = 0;
  
  std::vector<int> division_positions = {};



#if HAVE_CUDA
  cudaStream_t custreams[2];
  cudaStream_t pack_custream;
  cudaStream_t unpack_custream;
  cudaEvent_t pack_cuevent;
  cudaEvent_t unpack_cuevent;
#endif
// The CUDA stream used for data transfers and within-allreduce operations.
// A naive implementation would use the TensorFlow StreamExecutor CUDA
// stream. However, the allreduce and allgather require doing memory copies
// and kernel executions (for accumulation of values on the GPU). However,
// the subsequent operations must wait for those operations to complete,
// otherwise MPI (which uses its own stream internally) will begin the data
// transfers before the CUDA calls are complete. In order to wait for those
// CUDA operations, if we were using the TensorFlow stream, we would have to
// synchronize that stream; however, other TensorFlow threads may be
// submitting more work to that stream, so synchronizing on it can cause the
// allreduce to be delayed, waiting for compute totally unrelated to it in
// other parts of the graph. Overlaying memory transfers and compute during
// backpropagation is crucial for good performance, so we cannot use the
// TensorFlow stream, and must use our own stream.
#if HAVE_CUDA
  std::unordered_map<int, cudaStream_t> streams;
#endif
#if HAVE_NCCL
  std::unordered_map<std::vector<int32_t>, ncclComm_t> nccl_comms;
  std::unordered_map<std::vector<int32_t>, ncclComm_t> nccl_local_comms;
  std::unordered_map<std::vector<int32_t>, ncclComm_t> nccl_cross_comms;
#endif

  // Will be set to true after initialization when ddl is used
  bool ddl_initialized = false;
  int32_t ddl_local_device_id = 0;

// We reuse CUDA events as it appears that their creation carries non-zero cost.
// Event management code is only used in NCCL path.
#if HAVE_NCCL || HAVE_DDL
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex cuda_events_mutex;
#endif

  ~HorovodGlobalState() {
    // Make sure that the destructor of the background thread is safe to
    // call. If a thread is still joinable (not detached or complete) its
    // destructor cannot be called.
    if (background_thread.joinable()) {
      shut_down = true;
      background_thread.join();
    }
  }
};

// All the Horovod state that must be stored globally per-process.
HorovodGlobalState horovod_global;

// For clarify in argument lists.
#define RANK_ZERO 0

// Stall-check warning time
#define STALL_WARNING_TIME std::chrono::seconds(60)

const Status NOT_INITIALIZED_ERROR = Status::PreconditionError(
    "Horovod has not been initialized; use hvd.init().");

const Status SHUT_DOWN_ERROR = Status::Aborted(
    "Horovod has been shut down. This was caused by an exception on one of the "
    "ranks or an attempt to allreduce, allgather or broadcast a tensor after "
    "one of the ranks finished execution. If the shutdown was caused by an "
    "exception, you should see the exception in the log before the first "
    "shutdown message.");

#define OP_ERROR(entries, error_message)                                       \
  {                                                                            \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(error_message));                       \
      }                                                                        \
      return;                                                                  \
  }

std::shared_ptr<horovod::common::OpContext> CreateMXOpContext(int device);

// Store the MPIRequest for a name, and return whether the total count of
// MPIRequests for that tensor is now equal to the MPI size (and thus we are
// ready to reduce the tensor).
bool IncrementTensorCount(std::unique_ptr<MessageTable>& message_table,
                          MPIRequest msg, int mpi_size) {
  auto& name = msg.tensor_name();
  auto& timeline = horovod_global.timeline;
  auto table_iter = message_table->find(name);
  if (table_iter == message_table->end()) {
    std::vector<MPIRequest> messages = {msg};
    auto now = std::chrono::steady_clock::now();
    message_table->emplace(name, std::make_tuple(std::move(messages), now));
    table_iter = message_table->find(name);
    timeline.NegotiateStart(name, msg.request_type());
  } else {
    std::vector<MPIRequest>& messages = std::get<0>(table_iter->second);
    messages.push_back(msg);
  }

  timeline.NegotiateRankReady(name, msg.request_rank());

  std::vector<MPIRequest>& messages = std::get<0>(table_iter->second);
  int count = (int)messages.size();
  bool ready_to_reduce = count == mpi_size;
  if (ready_to_reduce) {
    timeline.NegotiateEnd(name);
  }
  return ready_to_reduce;
}

// Once a tensor is ready to be reduced, the coordinator sends an MPIResponse
// instructing all ranks to start the reduction to all ranks. The MPIResponse
// also contains error messages in case the submitted MPIRequests were not
// valid (for example, contained mismatched shapes or types).
//
// Constructing the MPIResponse, thus, requires a whole lot of error checking.
MPIResponse ConstructMPIResponse(std::unique_ptr<MessageTable>& message_table,
                                 std::string name) {
  bool error = false;
  auto it = message_table->find(name);
  assert(it != message_table->end());

  std::vector<MPIRequest>& requests = std::get<0>(it->second);
  assert(requests.size() > 0);

  std::ostringstream error_message_stream;

  // Check that all data types of tensors being reduced, gathered or broadcasted
  // are identical.
  auto data_type = requests[0].tensor_type();
  for (unsigned int i = 1; i < requests.size(); i++) {
    auto request_type = requests[i].tensor_type();
    if (data_type != request_type) {
      error = true;
      error_message_stream << "Mismatched data types: One rank had type "
                           << MPIDataType_Name(data_type)
                           << ", but another rank had type "
                           << MPIDataType_Name(request_type) << ".";
      break;
    }
  }

  // Check that all requested operations are the same
  auto message_type = requests[0].request_type();
  for (unsigned int i = 1; i < requests.size(); i++) {
    if (error) {
      break;
    }

    auto request_type = requests[i].request_type();
    if (message_type != request_type) {
      error = true;
      error_message_stream << "Mismatched MPI operations: One rank did an "
                           << MPIRequest::RequestType_Name(message_type)
                           << ", but another rank did an "
                           << MPIRequest::RequestType_Name(request_type) << ".";
      break;
    }
  }

  // If we are doing an allreduce or broadcast, check that all tensor shapes are
  // identical.
  if (message_type == MPIRequest::ALLREDUCE ||
      message_type == MPIRequest::BROADCAST) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }
    for (unsigned int i = 1; i < requests.size(); i++) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto dim : requests[i].tensor_shape()) {
        request_shape.AddDim(dim);
      }
      if (tensor_shape != request_shape) {
        error = true;
        error_message_stream
            << "Mismatched " << MPIRequest::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of shape "
            << tensor_shape.DebugString()
            << ", but another rank sent a tensor of shape "
            << request_shape.DebugString() << ".";
        break;
      }
    }
  }

  // If we are doing an allgather, make sure all but the first dimension are
  // the same. The first dimension may be different and the output tensor is
  // the sum of the first dimension. Collect the sizes by rank.
  std::vector<int64_t> tensor_sizes(requests.size());
  if (message_type == MPIRequest::ALLGATHER) {
    if (horovod_global.two_stage_loop) {
      error = true;
      error_message_stream << "Allgather not supported with HOROVOD_TWO_STAGE_LOOP=1. "
                           << " Disable this feature to run.";
    } else {
      TensorShape tensor_shape;
      for (auto dim : requests[0].tensor_shape()) {
        tensor_shape.AddDim(dim);
      }

      if (tensor_shape.dims() == 0) {
        error = true;
        error_message_stream << "Rank zero tried to "
                             << MPIRequest::RequestType_Name(message_type)
                             << " a rank-zero tensor.";
      } else {
        tensor_sizes[requests[0].request_rank()] = tensor_shape.dim_size(0);
      }

      for (unsigned int i = 1; i < requests.size(); i++) {
        if (error) {
          break;
        }

        TensorShape request_shape;
        for (auto dim : requests[i].tensor_shape()) {
          request_shape.AddDim(dim);
        }
        if (tensor_shape.dims() != request_shape.dims()) {
          error = true;
          error_message_stream
              << "Mismatched " << MPIRequest::RequestType_Name(message_type)
              << " tensor shapes: One rank sent a tensor of rank "
              << tensor_shape.dims()
              << ", but another rank sent a tensor of rank "
              << request_shape.dims() << ".";
          break;
        }

        bool dim_mismatch = false;
        for (int dim = 1; dim < tensor_shape.dims(); dim++) {
          if (tensor_shape.dim_size(dim) != request_shape.dim_size(dim)) {
            error = true;
            error_message_stream
                << "Mismatched " << MPIRequest::RequestType_Name(message_type)
                << " tensor shapes: One rank sent a tensor with dimension " << dim
                << " equal to " << tensor_shape.dim_size(dim)
                << ", but another rank sent a tensor with dimension " << dim
                << " equal to " << request_shape.dim_size(dim) << ".";
            dim_mismatch = true;
            break;
          }
        }
        if (dim_mismatch) {
          break;
        }

        tensor_sizes[requests[i].request_rank()] = request_shape.dim_size(0);
      }
    }
  }

  // If we are doing a broadcast, check that all root ranks are identical.
  if (message_type == MPIRequest::BROADCAST) {
    int first_root_rank = requests[0].root_rank();
    for (unsigned int i = 1; i < requests.size(); i++) {
      if (error) {
        break;
      }

      int this_root_rank = requests[i].root_rank();
      if (first_root_rank != this_root_rank) {
        error = true;
        error_message_stream
            << "Mismatched " << MPIRequest::RequestType_Name(message_type)
            << " root ranks: One rank specified root rank " << first_root_rank
            << ", but another rank specified root rank " << this_root_rank
            << ".";
        break;
      }
    }
  }

  bool first_device_is_cpu = requests[0].device() == CPU_DEVICE_ID;
  for (unsigned int i = 1; i < requests.size(); i++) {
    if (error) {
      break;
    }

    bool this_device_is_cpu = requests[i].device() == CPU_DEVICE_ID;
    if (first_device_is_cpu != this_device_is_cpu) {
      error = true;
      error_message_stream
          << "Mismatched " << MPIRequest::RequestType_Name(message_type)
          << " CPU/GPU device selection: One rank specified device "
          << (first_device_is_cpu ? "CPU" : "GPU")
          << ", but another rank specified device "
          << (this_device_is_cpu ? "CPU" : "GPU") << ".";
      break;
    }
  }

  std::vector<int32_t> devices;
  if (horovod_global.two_stage_loop || horovod_global.fixed_payload != 0) {
    devices.resize(1);
  } else {
    devices.resize(requests.size());
  }

  for (auto& request : requests) {
    if (horovod_global.two_stage_loop || horovod_global.fixed_payload != 0) {
      // Note: Device lists generated here aren't used for anything functional
      // and are currently restrictive.
      // Setting single list value to either CPU device or GPU device (0) when
      // using alternative paths.
      devices[0] = (request.device() == CPU_DEVICE_ID) ? CPU_DEVICE_ID : 0;
    } else {
      devices[request.request_rank()] = request.device();
    }
  }

  MPIResponse response;
  response.add_tensor_names(name);
  if (error) {
    std::string error_message = error_message_stream.str();
    response.set_response_type(MPIResponse::ERROR);
    response.set_error_message(error_message);
  } else if (message_type == MPIRequest::ALLGATHER) {
    response.set_response_type(MPIResponse::ALLGATHER);
    for (auto dim : tensor_sizes) {
      response.add_tensor_sizes(dim);
    }
  } else if (message_type == MPIRequest::ALLREDUCE) {
    response.set_response_type(MPIResponse::ALLREDUCE);
  } else if (message_type == MPIRequest::BROADCAST) {
    response.set_response_type(MPIResponse::BROADCAST);
  }
  response.set_devices(devices);

  // Clear all queued up requests for this name. They are now taken care of
  // by the constructed MPI response.
  message_table->erase(it);

  return response;
}

// Populates provided MPIResponseList with responses from map. Fuses allreduce
// responses by datatype when appropriate.
void PopulateMPIResponseList(MPIResponseList& response_list,
                             std::map<MPIDataType, std::deque<MPIResponse>>& responses_by_type,
                             HorovodGlobalState& state) {

  for (auto& res : responses_by_type) {
    auto& responses = res.second;
    while (!responses.empty()) {
      auto response = responses.front();
      assert(response.tensor_names().size() == 1);
      responses.pop_front();

      if (response.response_type() == MPIResponse::ResponseType::ALLREDUCE) {
        // Attempt to add more responses to this fused response.
        auto& entry = state.tensor_table[response.tensor_names()[0]];
        int64_t tensor_size = entry.tensor->size();

        while (!responses.empty()) {
          auto new_response = responses.front();
          assert(new_response.tensor_names().size() == 1);
          auto& new_entry = state.tensor_table[new_response.tensor_names()[0]];
          int64_t new_tensor_size = new_entry.tensor->size();

          if (response.response_type() == new_response.response_type() &&
              response.devices() == new_response.devices() &&
              entry.tensor->dtype() == new_entry.tensor->dtype() &&
              tensor_size + new_tensor_size <= state.tensor_fusion_threshold) {
            tensor_size += new_tensor_size;
            response.add_tensor_names(new_response.tensor_names()[0]);
            responses.pop_front();
          } else {
            // Don't try to fuse additional tensors since they are usually
            // computed in order of requests and skipping tensors may mean
            // that the batch will have to wait longer while skipped tensors
            // could be reduced at that time.
            break;
          }
        }
      }

      response_list.add_responses(response);
    }
  }
}

MPI_Datatype GetMPIDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_UINT8:
    return MPI_UINT8_T;
  case HOROVOD_INT8:
    return MPI_INT8_T;
  case HOROVOD_UINT16:
    return MPI_UINT16_T;
  case HOROVOD_INT16:
    return MPI_INT16_T;
  case HOROVOD_INT32:
    return MPI_INT32_T;
  case HOROVOD_INT64:
    return MPI_INT64_T;
  case HOROVOD_FLOAT16:
    return horovod_global.mpi_float16_t;
  case HOROVOD_FLOAT32:
    return MPI_FLOAT;
  case HOROVOD_FLOAT64:
    return MPI_DOUBLE;
  case HOROVOD_BOOL:
    return MPI_C_BOOL;
  default:
    throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
                           " is not supported in MPI mode.");
  }
}

size_t GetDataTypeSize(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_UINT8:
    return sizeof(unsigned char);
  case HOROVOD_INT8:
    return sizeof(char);
  case HOROVOD_UINT16:
    return sizeof(unsigned short int);
  case HOROVOD_INT16:
    return sizeof (short int);
  case HOROVOD_INT32:
    return sizeof(int);
  case HOROVOD_INT64:
    return sizeof(long long int);
  case HOROVOD_FLOAT16:
    return sizeof(short int);
  case HOROVOD_FLOAT32:
    return sizeof(float);
  case HOROVOD_FLOAT64:
    return sizeof(double);
  case HOROVOD_BOOL:
    return sizeof(bool);
  default:
    throw std::logic_error("Cannot get size of type " + MPIDataType_Name(tensor->dtype()));
  }
}

#if HAVE_NCCL
ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_INT32:
    return ncclInt32;
  case HOROVOD_INT64:
    return ncclInt64;
  case HOROVOD_FLOAT16:
    return ncclFloat16;
  case HOROVOD_FLOAT32:
    return ncclFloat32;
  case HOROVOD_FLOAT64:
    return ncclFloat64;
  default:
    throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
                           " is not supported in NCCL mode.");
  }
}
#endif

#if HAVE_DDL
DDL_Type GetDDLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_FLOAT32:
    return DDL_TYPE_FLOAT;
  default:
    throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
                           " is not supported in DDL mode.");
  }
}
#endif

#define MPI_CHECK(entries, op_name, op)                                        \
  {                                                                            \
    auto mpi_result = (op);                                                    \
    if (mpi_result != MPI_SUCCESS) {                                           \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(                                       \
            std::string(op_name) + " failed, see MPI output for details."));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define CUDA_CHECK(entries, op_name, op)                                       \
  {                                                                            \
    auto cuda_result = (op);                                                   \
    if (cuda_result != cudaSuccess) {                                          \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(std::string(op_name) + " failed: " +   \
                                        cudaGetErrorString(cuda_result)));     \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define NCCL_CHECK(entries, op_name, op)                                       \
  {                                                                            \
    auto nccl_result = (op);                                                   \
    if (nccl_result != ncclSuccess) {                                          \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(std::string(op_name) + " failed: " +   \
                                        ncclGetErrorString(nccl_result)));     \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define DDL_CHECK(entries, op_name, op)                                        \
  {                                                                            \
    auto ddl_result = (op);                                                    \
    if (ddl_result != DDL_SUCCESS) {                                           \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(std::string(op_name) + " failed."));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

// This event management code is only used in NCCL.
#if HAVE_NCCL || HAVE_DDL
cudaError_t GetCudaEvent(cudaEvent_t* event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = horovod_global.cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto& queue = horovod_global.cuda_events[device];
    if (!queue.empty()) {
      *event = queue.front();
      queue.pop();
      return cudaSuccess;
    }
  }

  return cudaEventCreateWithFlags(event, cudaEventBlockingSync |
                                             cudaEventDisableTiming);
}

cudaError_t ReleaseCudaEvent(cudaEvent_t event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = horovod_global.cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto& queue = horovod_global.cuda_events[device];
    queue.push(event);
  }

  return cudaSuccess;
}

#define RECORD_EVENT(entries, event_queue, name, stream)                       \
  {                                                                            \
    cudaEvent_t event;                                                         \
    CUDA_CHECK(entries, "GetCudaEvent", GetCudaEvent(&event))                  \
    CUDA_CHECK(entries, "cudaEventRecord", cudaEventRecord(event, stream))     \
    (event_queue).emplace(name, event);                                        \
  }

#define WAIT_FOR_EVENTS(entries, timeline, event_queue)                        \
  {                                                                            \
    while (!(event_queue).empty()) {                                           \
      std::string name;                                                        \
      cudaEvent_t event;                                                       \
      std::tie(name, event) = (event_queue).front();                           \
      (event_queue).pop();                                                     \
      if (name != "") {                                                        \
        ACTIVITY_START_ALL(entries, timeline, name)                            \
      }                                                                        \
      CUDA_CHECK(entries, "cudaEventSynchronize", cudaEventSynchronize(event)) \
      if (name != "") {                                                        \
        ACTIVITY_END_ALL(entries, timeline)                                    \
      }                                                                        \
      CUDA_CHECK(entries, "ReleaseCudaEvent", ReleaseCudaEvent(event))         \
    }                                                                          \
  }
#endif

#define ACTIVITY_START_ALL(entries, timeline, activity)                        \
  {                                                                            \
    for (auto& e : (entries)) {                                                \
      (timeline).ActivityStart(e.tensor_name, activity);                       \
    }                                                                          \
  }

#define ACTIVITY_END_ALL(entries, timeline)                                    \
  {                                                                            \
    for (auto& e : (entries)) {                                                \
      (timeline).ActivityEnd(e.tensor_name);                                   \
    }                                                                          \
  }

#define CUDA_CHECK_STATUS(op_name, op)                                \
  {                                                                            \
    auto cuda_result = (op);                                                   \
    if (cuda_result != cudaSuccess) {                                          \
      return Status::UnknownError(std::string(op_name) + " failed: " +         \
                                  cudaGetErrorString(cuda_result));            \
    }                                                                          \
  }

#define NCCL_CHECK_STATUS(op_name, op)                                         \
  {                                                                            \
    auto nccl_result = (op);                                                   \
    if (nccl_result != ncclSuccess) {                                          \
      return Status::UnknownError(std::string(op_name) + " failed: " +         \
                                  ncclGetErrorString(nccl_result));            \
    }                                                                          \
  }

#define MPI_CHECK_STATUS(op_name, op)                                          \
  {                                                                            \
    auto mpi_result = (op);                                                    \
    if (mpi_result != MPI_SUCCESS) {                                           \
      return Status::UnknownError(                                             \
         std::string(op_name) + " failed, see MPI output for details.");       \
    }                                                                          \
  }

Status AllocateBuffer(std::shared_ptr<OpContext>& context,
                      std::shared_ptr<PersistentBuffer>& buffer) {
      // Lazily allocate persistent buffer for Tensor Fusion and keep it
      // forever per device.
      size_t buf_size = horovod_global.tensor_fusion_threshold;

      // Add padding to allocation for allreduce_mode = 1 (hierarchical on GPU).
      // Need a max of ALIGN_BYTES * local_size padding to guarantee enough space.
      if (horovod_global.allreduce_mode == 1) buf_size += ALIGN_BYTES * horovod_global.local_size;

      Status status = context->AllocatePersistent(
          buf_size, &buffer);
      if (!status.ok()) {
        return status;
      }
#if HAVE_CUDA
      if (!horovod_global.pack_ptrs.allocated) {
        //CUDA_CHECK(entries, "cudaSetDevice", cudaSetDevice(first_entry.device))
        CUDA_CHECK_STATUS("cudaSetDevice", cudaSetDevice(horovod_global.local_rank))
        CUDA_CHECK_STATUS("cudaMallocHost",
                   cudaMallocHost(&horovod_global.pack_ptrs.pack_out, PACK_PTRS_CAPACITY*sizeof(float*)))
        CUDA_CHECK_STATUS("cudaMallocHost",
                   cudaMallocHost(&horovod_global.pack_ptrs.pack_in, PACK_PTRS_CAPACITY*sizeof(float*)))
        CUDA_CHECK_STATUS("cudaMallocHost",
                   cudaMallocHost(&horovod_global.pack_ptrs.pack_sizes, PACK_PTRS_CAPACITY*sizeof(size_t)))
        CUDA_CHECK_STATUS("cudaMallocHost",
                   cudaMallocHost(&horovod_global.pack_ptrs.unpack_out, PACK_PTRS_CAPACITY*sizeof(float*)))
        CUDA_CHECK_STATUS("cudaMallocHost",
                   cudaMallocHost(&horovod_global.pack_ptrs.unpack_in, PACK_PTRS_CAPACITY*sizeof(float*)))
        CUDA_CHECK_STATUS("cudaMallocHost",
                   cudaMallocHost(&horovod_global.pack_ptrs.unpack_sizes, PACK_PTRS_CAPACITY*sizeof(size_t)))
        horovod_global.pack_ptrs.allocated = true;
      }
#endif
      return Status::OK();
}

Status InitializeNcclCommunicatorMode0(ncclComm_t& nccl_comm) {
  int nccl_rank, nccl_size;
  MPI_Comm nccl_id_bcast_comm;
  if (horovod_global.hierarchical_allreduce) {
    nccl_rank = horovod_global.local_rank;
    nccl_size = horovod_global.local_size;
    nccl_id_bcast_comm = horovod_global.local_comm;
  } else {
    nccl_rank = horovod_global.rank;
    nccl_size = horovod_global.size;
    nccl_id_bcast_comm = horovod_global.mpi_comm;
  }

  ncclUniqueId nccl_id;
  if (nccl_rank == 0) {
    NCCL_CHECK_STATUS("ncclGetUniqueId", ncclGetUniqueId(&nccl_id))
  }


  MPI_CHECK_STATUS("MPI_Bcast",
                   MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
		   nccl_id_bcast_comm));

  ncclComm_t new_nccl_comm;
  NCCL_CHECK_STATUS("ncclCommInitRank",
                    ncclCommInitRank(&new_nccl_comm, nccl_size, nccl_id, nccl_rank))
  nccl_comm = new_nccl_comm;

  // Barrier helps NCCL to synchronize after initialization and avoid
  // deadlock that we've been seeing without it.
  MPI_CHECK_STATUS("MPI_Barrier", MPI_Barrier(horovod_global.mpi_comm));

  return Status::OK();
}


// Process an MPIResponse by doing a reduction, a gather, a broadcast, or
// raising an error.
void PerformOperation(TensorTable& tensor_table, MPIResponse response) {
  std::vector<TensorTableEntry> entries;
  {
    // Lock on the tensor table.
    std::lock_guard<std::mutex> guard(horovod_global.mutex);

    for (auto& name : response.tensor_names()) {
      // We should never fail at finding this key in the tensor table.
      auto iter = tensor_table.find(name);
      assert(iter != tensor_table.end());

      assert(response.response_type() == MPIResponse::ALLREDUCE ||
             response.response_type() == MPIResponse::ALLGATHER ||
             response.response_type() == MPIResponse::BROADCAST ||
             response.response_type() == MPIResponse::ERROR);

      entries.push_back(iter->second);

      // Clear the tensor table of this tensor and its callbacks; the rest of
      // this function takes care of it.
      tensor_table.erase(iter);
    }
  }

  auto& timeline = horovod_global.timeline;
  for (auto& e : entries) {
    timeline.Start(e.tensor_name, response.response_type());
  }

  if (entries.size() > 0) {
    auto first_entry = entries[0];
    // Note: it is OK for different entries to come from different frameworks
    // since buffer allocated here is guaranteed to survive at least till the
    // end of this operation.
    auto& buffer = horovod_global.tensor_fusion_buffers[std::make_tuple(
        first_entry.device, first_entry.context->framework())];
    if (buffer == nullptr) {
      ACTIVITY_START_ALL(entries, timeline, INIT_FUSION_BUFFER)

      Status status = AllocateBuffer(first_entry.context, buffer);
      if (!status.ok()) {
        for (auto& e : entries) {
          timeline.End(e.tensor_name, nullptr);
          e.callback(status);
        }
        return;
      }

      ACTIVITY_END_ALL(entries, timeline)
    }
  }

  // On GPU data readiness is signalled by ready_event.
  std::vector<TensorTableEntry> waiting_tensors;
  for (auto& e : entries) {
    if (e.ready_event != nullptr) { // change status
      timeline.ActivityStart(e.tensor_name, WAIT_FOR_DATA); // status is waiting for data
      waiting_tensors.push_back(e);
    }
  }
  while (!waiting_tensors.empty()) { // waiting for previous tensor operations ?
    for (auto it = waiting_tensors.begin(); it != waiting_tensors.end();) {
      if (it->ready_event->Ready()) {
        timeline.ActivityEnd(it->tensor_name);
        timeline.ActivityStart(it->tensor_name, WAIT_FOR_OTHER_TENSOR_DATA); // change status
        it = waiting_tensors.erase(it);
      } else {
        ++it;
      }
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
  }
  for (auto& e : entries) {
    if (e.ready_event != nullptr) {
      timeline.ActivityEnd(e.tensor_name); // status is TOP_LEVEL
    }
  }

  Status status;
  if (response.response_type() == MPIResponse::ALLGATHER) {
    assert(entries.size() == 1);
    auto e = entries[0];

    // Copy tensor sizes from the MPI response into a vector of int64_t
    // and compute total size.  This is size of first dimension.
    std::vector<int64_t> tensor_sizes;
    int64_t total_dimension_size = 0;
    for (auto sz : response.tensor_sizes()) {
      tensor_sizes.push_back(sz);
      total_dimension_size += sz;
    }

    // Every tensor participating in Allgather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }

    // Allgather output will have shape of:
    // (sum of first dimension of every tensor) x (tensor slice shape).
    TensorShape output_shape;
    output_shape.AddDim((int64_t)total_dimension_size);
    output_shape.AppendShape(single_slice_shape);

    ACTIVITY_START_ALL(entries, timeline, ALLOCATE_OUTPUT)
    status = e.context->AllocateOutput(output_shape, &e.output);
    if (!status.ok()) {
      timeline.End(e.tensor_name, nullptr);
      e.callback(status);
      return;
    }
    ACTIVITY_END_ALL(entries, timeline)

    // Tensors may have different first dimension, so we need to use
    // MPI_Allgatherv API that supports gathering arrays of different length.
    ACTIVITY_START_ALL(entries, timeline, MPI_ALLGATHER)
    auto* recvcounts = new int[tensor_sizes.size()];
    auto* displcmnts = new int[tensor_sizes.size()];
    for (unsigned int i = 0; i < tensor_sizes.size(); i++) {
      recvcounts[i] =
          (int)(single_slice_shape.num_elements() * tensor_sizes[i]);
      if (i == 0) {
        displcmnts[i] = 0;
      } else {
        displcmnts[i] = recvcounts[i - 1] + displcmnts[i - 1];
      }
    }
    auto result = MPI_Allgatherv(
        e.tensor->data(), (int)e.tensor->shape().num_elements(),
        GetMPIDataType(e.tensor), (void*)e.output->data(), recvcounts,
        displcmnts, GetMPIDataType(e.tensor), horovod_global.mpi_comm);
    delete[] recvcounts;
    delete[] displcmnts;
    MPI_CHECK(entries, "MPI_Allgatherv", result)
    ACTIVITY_END_ALL(entries, timeline)

    timeline.End(e.tensor_name, e.output);
    e.callback(Status::OK());

  } else if (response.response_type() == MPIResponse::ALLREDUCE) {
    // When MPIResponse is ALLREDUCE, the following operations are executed

    auto& first_entry = entries[0];

#if HAVE_CUDA
    bool on_gpu = first_entry.device != CPU_DEVICE_ID;
    //Set GPU ID and create the greatest priority stream.
    if (on_gpu) {
      CUDA_CHECK(entries, "cudaSetDevice", cudaSetDevice(first_entry.device))

      // Ensure stream is in the map before executing reduction.
      cudaStream_t& stream = horovod_global.streams[first_entry.device];
      if (stream == nullptr) {
        int greatest_priority;
        // Original code
        CUDA_CHECK(entries, "cudaDeviceGetStreamPriorityRange",
                   cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority))
        CUDA_CHECK(entries, "cudaStreamCreateWithPriority",
                   cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                                greatest_priority))

        if( horovod_global.triple_streams4allreduce ){
          // Create stream and event for data packing on GPU memory
          CUDA_CHECK(entries, "cudaStreamCreateWithPriority",
                   cudaStreamCreateWithPriority(&horovod_global.pack_custream, cudaStreamNonBlocking,
                                                greatest_priority))
          CUDA_CHECK(entries, "cudaEventCreate",
                   cudaEventCreate(&horovod_global.pack_cuevent))

          // Create stream and event for data unpacking on GPU memory
          CUDA_CHECK(entries, "cudaStreamCreateWithPriority",
                   cudaStreamCreateWithPriority(&horovod_global.unpack_custream, cudaStreamNonBlocking,
                                                greatest_priority))
          CUDA_CHECK(entries, "cudaEventCreate",
                   cudaEventCreate(&horovod_global.unpack_cuevent))
        }
      }
    }
#endif

#if HOROVOD_GPU_ALLREDUCE=='N' || HOROVOD_GPU_ALLREDUCE=='D'// 'N' stands for NCCL
    if (on_gpu) {
      auto stream = horovod_global.streams[first_entry.device];
      auto event_queue = std::queue<std::pair<std::string, cudaEvent_t>>();

      // Determine GPU IDs of the devices participating in this communicator.
      std::vector<int32_t> nccl_device_map;
      if (horovod_global.hierarchical_allreduce &&
          !(horovod_global.two_stage_loop || horovod_global.fixed_payload != 0)) {
        for (int rank : horovod_global.local_comm_ranks) {
          nccl_device_map.push_back(response.devices()[rank]);
        }
      } else {
        nccl_device_map = response.devices();
      }

#if HOROVOD_GPU_ALLREDUCE=='N'
      std::vector<int32_t> gpu_device_id( horovod_global.local_rank );

      // Ensure NCCL communicator is in the map before executing reduction.
      //ncclComm_t& nccl_comm = horovod_global.nccl_comms[nccl_device_map];
      //ncclComm_t& nccl_local_comm = horovod_global.nccl_local_comms[nccl_device_map];
      //ncclComm_t& nccl_cross_comm = horovod_global.nccl_cross_comms[nccl_device_map];

      ncclComm_t& nccl_comm = horovod_global.nccl_comms[gpu_device_id];
      ncclComm_t& nccl_local_comm = horovod_global.nccl_local_comms[gpu_device_id];
      ncclComm_t& nccl_cross_comm = horovod_global.nccl_cross_comms[gpu_device_id];


      // Initialize global NCCL communicator
      if ((horovod_global.allreduce_mode == 0 || first_entry.tensor->size() >
           horovod_global.tensor_fusion_threshold) && nccl_comm == nullptr) {

        ACTIVITY_START_ALL(entries, timeline, INIT_NCCL) // chage status of all entries in "entries" to "activity"

        Status status = InitializeNcclCommunicatorMode0(nccl_comm);
        if (!status.ok()) {
          for (auto& e : entries) {
            timeline.End(e.tensor_name, nullptr);
            e.callback(status);
          }
        }

        ACTIVITY_END_ALL(entries, timeline)

      }

      // Iniitialize local and cross NCCL communicators
      if (horovod_global.allreduce_mode == 1 && nccl_local_comm == nullptr &&
                 nccl_cross_comm == nullptr) {

        ACTIVITY_START_ALL(entries, timeline, INIT_NCCL)
        ncclUniqueId nccl_id;
        if (horovod_global.local_rank == 0) {
          NCCL_CHECK(entries, "ncclGetUniqueId", ncclGetUniqueId(&nccl_id))
        }

        MPI_CHECK(entries, "MPI_Bcast",
                  MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                            horovod_global.local_comm));

        ncclComm_t new_nccl_local_comm;
        NCCL_CHECK(
            entries, "ncclCommInitRank",
            ncclCommInitRank(&new_nccl_local_comm, horovod_global.local_size, nccl_id, horovod_global.local_rank))

        nccl_local_comm = new_nccl_local_comm;

        MPI_CHECK(entries, "MPI_Barrier", MPI_Barrier(horovod_global.local_comm));


        if (horovod_global.rank < horovod_global.local_size) {
          NCCL_CHECK(entries, "ncclGetUniqueId", ncclGetUniqueId(&nccl_id))
        }

        MPI_CHECK(entries, "MPI_Bcast",
                  MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                            horovod_global.cross_comm));

        ncclComm_t new_nccl_cross_comm;
        NCCL_CHECK(
            entries, "ncclCommInitRank",
            ncclCommInitRank(&new_nccl_cross_comm, horovod_global.cross_size, nccl_id, horovod_global.cross_rank))
        nccl_cross_comm = new_nccl_cross_comm;

        MPI_CHECK(entries, "MPI_Barrier", MPI_Barrier(horovod_global.cross_comm));

        // Barrier helps NCCL to synchronize after initialization and avoid
        // deadlock that we've been seeing without it.
        MPI_CHECK(entries, "MPI_Barrier", MPI_Barrier(horovod_global.mpi_comm));

        ACTIVITY_END_ALL(entries, timeline)
      }

#elif HOROVOD_GPU_ALLREDUCE == 'D'
      if (!horovod_global.ddl_initialized) {
        // Initialize DDL
        auto ddl_options = std::getenv("DDL_OPTIONS");
        if (ddl_options == nullptr) {
          OP_ERROR(entries, "DDL_OPTIONS env variable needs to be set to use DDL.")
        }
        DDL_CHECK(entries, "ddl_init", ddl_init(ddl_options))
        horovod_global.ddl_initialized = true;
        horovod_global.ddl_local_device_id = first_entry.device;
      } else if (horovod_global.ddl_local_device_id != first_entry.device) {
        OP_ERROR(entries, "DDL does not support more than one GPU device per process.")
      }
#endif

      if (timeline.Initialized()) {
        RECORD_EVENT(entries, event_queue, QUEUE, stream)
      }

      // If entries.size() > 1, we copy tensors into fusion buffer before
      // allreduce, and distribute results of allreduce back into target
      // tensors after allreduce.
      // If there is a single entry and it will fit, also copy to fusion buffer.

      const void* fused_input_data;
      void* buffer_data;
      int64_t num_elements = 0;
      size_t buffer_len;

      if (entries.size() > 1 || first_entry.output->size() <= horovod_global.tensor_fusion_threshold) {
        // Access the fusion buffer.
        auto& buffer = horovod_global.tensor_fusion_buffers[std::make_tuple(
            first_entry.device, first_entry.context->framework())];
        buffer_data =
            const_cast<void*>(buffer->AccessData(first_entry.context));

        // Copy memory into the fusion buffer.
        if (entries.size() <= PACK_PTRS_CAPACITY) {
          int64_t offset = ( horovod_global.triple_streams4allreduce) ? 
                    horovod_global.transfer_buffer_offset : 0 ;
          int idx = horovod_global.num_transferred_layer;
 
          void **pack_out_tmp, **pack_in_tmp;
          size_t *pack_sizes_tmp;
          pack_out_tmp = &(horovod_global.pack_ptrs.pack_out[idx]);
          pack_in_tmp = &(horovod_global.pack_ptrs.pack_in[idx]);
          pack_sizes_tmp = &(horovod_global.pack_ptrs.pack_sizes[idx]);

          // Set input/output pointers and sizes
          for (auto& e : entries) {
            void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

            horovod_global.pack_ptrs.pack_out[idx] = buffer_data_at_offset;
            horovod_global.pack_ptrs.pack_in[idx] = (void*) e.tensor->data();
            horovod_global.pack_ptrs.pack_sizes[idx] = e.tensor->size();

            offset += e.tensor->size();
            idx++;
          }
          buffer_len = (size_t)offset;

          // Perform batched d2d memcpy
          if( horovod_global.triple_streams4allreduce ){
            batched_d2d_memcpy(pack_out_tmp,
                               pack_in_tmp,
                               pack_sizes_tmp,
                               entries.size(),
                               horovod_global.pack_custream);


          } else {
            batched_d2d_memcpy(pack_out_tmp,
                               pack_in_tmp,
                               pack_sizes_tmp,
                               entries.size(),
                               stream);
          }
        } else {
          int64_t offset = 0;
          for (auto& e : entries) {
            void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
            CUDA_CHECK(entries, "cudaMemcpyAsync",
                       cudaMemcpyAsync(buffer_data_at_offset, e.tensor->data(),
                                       (size_t)e.tensor->size(),
                                       cudaMemcpyDeviceToDevice, stream))
            offset += e.tensor->size();
          }
          buffer_len = (size_t)offset;
        }
        if (timeline.Initialized() || horovod_global.ddl_initialized) {
          RECORD_EVENT(entries, event_queue, MEMCPY_IN_FUSION_BUFFER, stream)
        }

        // Set the input data to originate from the buffer.
        fused_input_data = buffer_data;
        if( horovod_global.triple_streams4allreduce ){
          fused_input_data = (uint8_t*)fused_input_data + horovod_global.transfer_buffer_offset;
        }

        // Perform the reduction on the fusion buffer.
        for (auto& e : entries) {
          num_elements += e.tensor->shape().num_elements();
        }

      } else {
        fused_input_data = first_entry.tensor->data();
        buffer_data = (void*)first_entry.output->data();
        num_elements = first_entry.tensor->shape().num_elements();
        buffer_len = (size_t)first_entry.output->size();
        if (horovod_global.ddl_initialized) {
          // Copy input buffer content to output buffer
          // because DDL only supports in-place allreduce
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(buffer_data, fused_input_data,
                                     buffer_len,
                                     cudaMemcpyDeviceToDevice, stream))
          RECORD_EVENT(entries, event_queue, MEMCPY_IN_FUSION_BUFFER, stream)
        }
      }

      void* host_buffer = nullptr;
#if HOROVOD_GPU_ALLREDUCE == 'D'
      // Synchronize.
      WAIT_FOR_EVENTS(entries, timeline, event_queue)
      DDL_Type ddl_data_type;
      try {
        ddl_data_type = GetDDLDataType(first_entry.tensor);
      } catch (const std::logic_error& ex) {
        OP_ERROR(entries, ex.what())
      }
      DDL_CHECK(entries, "ddl_allreduce",
                ddl_allreduce(buffer_data,
                              (size_t)num_elements,
                              ddl_data_type,
                              DDL_OP_SUM))
#else
      if (horovod_global.hierarchical_allreduce) {
        NCCL_CHECK(entries, "ncclReduce",
                   ncclReduce(fused_input_data, buffer_data,
                              (size_t)num_elements,
                              GetNCCLDataType(first_entry.tensor), ncclSum, 0,
                              nccl_comm, stream))
        if (timeline.Initialized()) {
          RECORD_EVENT(entries, event_queue, NCCL_REDUCE, stream)
        }

        if (horovod_global.local_rank == 0) {
          // cudaHostAlloc is significantly slower than malloc.  Pre-allocating
          // a buffer is not safe since the tensor can be arbitrarily large.
          host_buffer = malloc(buffer_len);

          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(host_buffer, buffer_data, buffer_len,
                                     cudaMemcpyDeviceToHost, stream))
          // This event must be recorded for the subsequent synchronize.
          RECORD_EVENT(entries, event_queue, MEMCPY_IN_HOST_BUFFER, stream)

          // Synchronize.
          WAIT_FOR_EVENTS(entries, timeline, event_queue)

          ACTIVITY_START_ALL(entries, timeline, MPI_ALLREDUCE)
          MPI_CHECK(entries, "MPI_Allreduce",
                    MPI_Allreduce(MPI_IN_PLACE, host_buffer, (int)num_elements,
                                  GetMPIDataType(first_entry.tensor), MPI_SUM,
                                  horovod_global.cross_comm))
          ACTIVITY_END_ALL(entries, timeline)

          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(buffer_data, host_buffer, buffer_len,
                                     cudaMemcpyHostToDevice, stream))
          if (timeline.Initialized()) {
            RECORD_EVENT(entries, event_queue, MEMCPY_OUT_HOST_BUFFER, stream)
          }
        }

        NCCL_CHECK(entries, "ncclBcast",
                   ncclBcast(buffer_data, (size_t)num_elements,
                             GetNCCLDataType(first_entry.tensor), 0, nccl_comm,
                             stream))
        if (timeline.Initialized()) {
          RECORD_EVENT(entries, event_queue, NCCL_BCAST, stream)
        }
      } else {

        size_t num_elements_per_rank = 0;
        if (horovod_global.allreduce_mode == 1) {
          num_elements_per_rank = (num_elements + horovod_global.local_size - 1) / horovod_global.local_size;
          // align buffers to ALIGN_BYTES bytes
          int align = ALIGN_BYTES / GetDataTypeSize(first_entry.tensor);
          num_elements_per_rank = (num_elements_per_rank + align - 1) / align * align;
        }

        if (horovod_global.allreduce_mode == 0 ||
            first_entry.tensor->size() > horovod_global.tensor_fusion_threshold) {

          if( horovod_global.triple_streams4allreduce ){
            cudaEventRecord( horovod_global.pack_cuevent, horovod_global.pack_custream );
            cudaStreamWaitEvent( stream, horovod_global.pack_cuevent, 0 );
          }

          void* buffer_temp_pointer = (uint8_t*)buffer_data;
          if( horovod_global.triple_streams4allreduce )
            buffer_temp_pointer = (uint8_t*)buffer_temp_pointer + horovod_global.transfer_buffer_offset;

          NCCL_CHECK(entries, "ncclAllReduce",
                     ncclAllReduce(fused_input_data, buffer_temp_pointer/*buffer_data*/,
                                   (size_t)num_elements,
                                   GetNCCLDataType(first_entry.tensor), ncclSum,
                                   nccl_comm, stream))

          if( horovod_global.triple_streams4allreduce ){
            cudaEventRecord( horovod_global.unpack_cuevent, stream );
            cudaStreamWaitEvent( horovod_global.unpack_custream, horovod_global.unpack_cuevent, 0 );
          }

        } else if (horovod_global.allreduce_mode == 1) {

          // When HOROVOD_ALLREDUCE_MODE is enable, the following operations are executed as a hierarchical allreduce.          
          void* buffer_temp_pointer = (uint8_t*)buffer_data;

          if( horovod_global.triple_streams4allreduce )
            buffer_temp_pointer = (uint8_t*)buffer_temp_pointer + horovod_global.transfer_buffer_offset;

          auto buffer_at_offset = (uint8_t*)buffer_temp_pointer + num_elements_per_rank * 
                      GetDataTypeSize(first_entry.tensor) * horovod_global.local_rank;

          // MPI Barrier before nccl allreduce
          if( horovod_global.multiple_transfer_mode ){
            MPI_Barrier(horovod_global.local_comm);
          }

          if( horovod_global.triple_streams4allreduce ){
            cudaEventRecord( horovod_global.pack_cuevent, horovod_global.pack_custream );
            cudaStreamWaitEvent( stream, horovod_global.pack_cuevent, 0 );
          }
          NCCL_CHECK(entries, "ncclReduceScatter",
                     ncclReduceScatter(fused_input_data, buffer_at_offset,
                                       (size_t) num_elements_per_rank,
                                       GetNCCLDataType(first_entry.tensor), ncclSum,
                                       nccl_local_comm, stream))
          NCCL_CHECK(entries, "ncclAllReduce",
                     ncclAllReduce(buffer_at_offset, buffer_at_offset,
                                   (size_t)num_elements_per_rank,
                                   GetNCCLDataType(first_entry.tensor), ncclSum,
                                   nccl_cross_comm, stream))
          NCCL_CHECK(entries, "ncclAllGather",
                     ncclAllGather(buffer_at_offset, buffer_temp_pointer/*buffer_data*/,
                                   (size_t)num_elements_per_rank,
                                   GetNCCLDataType(first_entry.tensor),
                                   nccl_local_comm, stream))
          if( horovod_global.triple_streams4allreduce ){
            cudaEventRecord( horovod_global.unpack_cuevent, stream );
            cudaStreamWaitEvent( horovod_global.unpack_custream, horovod_global.unpack_cuevent, 0 );

          }
        }
      }
#endif
      if (timeline.Initialized()) {
        RECORD_EVENT(entries, event_queue, NCCL_ALLREDUCE, stream)
      }

      if (entries.size() > 1 || first_entry.output->size() <= horovod_global.tensor_fusion_threshold) {
        // Copy memory out of the fusion buffer.
        if (entries.size() <= PACK_PTRS_CAPACITY) {
          int64_t offset = ( horovod_global.triple_streams4allreduce) ? 
                    horovod_global.transfer_buffer_offset : 0 ;
          int idx = horovod_global.num_transferred_layer; 
          void **unpack_out_tmp, **unpack_in_tmp;
          size_t *unpack_sizes_tmp;
          unpack_out_tmp = &(horovod_global.pack_ptrs.unpack_out[idx]);
          unpack_in_tmp = &(horovod_global.pack_ptrs.unpack_in[idx]);
          unpack_sizes_tmp = &(horovod_global.pack_ptrs.unpack_sizes[idx]);

          // Set input/output pointers and sizes
          for (auto& e : entries) {
            void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

            horovod_global.pack_ptrs.unpack_out[idx] = (void*)(e.output->data());
            horovod_global.pack_ptrs.unpack_in[idx] = buffer_data_at_offset;
            horovod_global.pack_ptrs.unpack_sizes[idx] = e.tensor->size();

            offset += e.tensor->size();
            idx++;
          }

          horovod_global.num_transferred_layer = idx;
          if ( horovod_global.triple_streams4allreduce ){
            horovod_global.transfer_buffer_offset = (offset + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
            stream = horovod_global.unpack_custream;
          }

          // Perform batched d2d memcpy
          batched_d2d_memcpy(unpack_out_tmp,
                              unpack_in_tmp,
                              unpack_sizes_tmp,
                              entries.size(),
                              stream);

          // Sync here is required to ensure pack/unpack pointer for batch D2D memcpy
          // do not get overwritten by possible future iteration.

          if( horovod_global.multiple_transfer_mode ){
            if( horovod_global.num_transferred_layer == horovod_global.fixed_payload ){
              CUDA_CHECK(entries, "cudaStreamSynchronize", cudaStreamSynchronize(stream))
              horovod_global.num_transferred_layer = 0;
              horovod_global.transfer_buffer_offset = 0;
            }
          } else {
            CUDA_CHECK(entries, "cudaStreamSynchronize", cudaStreamSynchronize(stream))
            horovod_global.num_transferred_layer = 0;
            horovod_global.transfer_buffer_offset = 0;
          }
        } else {
          int64_t offset = 0;
          for (auto& e : entries) {
            void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
            CUDA_CHECK(entries, "cudaMemcpyAsync",
                       cudaMemcpyAsync((void*)e.output->data(),
                                       buffer_data_at_offset,
                                       (size_t)e.tensor->size(),
                                       cudaMemcpyDeviceToDevice, stream))
            offset += e.tensor->size();
          }
        }

        if (timeline.Initialized()) {
          RECORD_EVENT(entries, event_queue, MEMCPY_OUT_FUSION_BUFFER, stream)
        }
      }

      // Use completion marker via event because it's faster than
      // blocking cudaStreamSynchronize() in this thread.
      RECORD_EVENT(entries, event_queue, "", stream)
      

      // TODO: use thread pool or single thread for callbacks
      std::thread finalizer_thread([entries, first_entry, host_buffer, response,
                                    event_queue, &timeline]() mutable {
        CUDA_CHECK(entries, "cudaSetDevice", cudaSetDevice(first_entry.device))

        WAIT_FOR_EVENTS(entries, timeline, event_queue)

        if (host_buffer != nullptr) {
          free(host_buffer);
        }

        for (auto& e : entries) {
          timeline.End(e.tensor_name, e.output);
          e.callback(Status::OK());
        }
      });
      finalizer_thread.detach();
      return;
    }
#endif

    if (entries.size() > 1) {
      // Access the fusion buffer.
      auto& buffer = horovod_global.tensor_fusion_buffers[std::make_tuple(
          first_entry.device, first_entry.context->framework())];
      auto buffer_data = buffer->AccessData(first_entry.context);

      // Copy memory into the fusion buffer.
      ACTIVITY_START_ALL(entries, timeline, MEMCPY_IN_FUSION_BUFFER)
      int64_t offset = 0;
      for (auto& e : entries) {
        void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
#if HAVE_CUDA
        if (on_gpu) {
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(
                         buffer_data_at_offset, e.tensor->data(),
                         (size_t)e.tensor->size(), cudaMemcpyDeviceToDevice,
                         horovod_global.streams[first_entry.device]))
        } else {
#endif
          std::memcpy(buffer_data_at_offset, e.tensor->data(),
                      (size_t)e.tensor->size());
#if HAVE_CUDA
        }
#endif
        offset += e.tensor->size();
      }
#if HAVE_CUDA
      if (on_gpu) {
        CUDA_CHECK(
            entries, "cudaStreamSynchronize",
            cudaStreamSynchronize(horovod_global.streams[first_entry.device]))
      }
#endif
      ACTIVITY_END_ALL(entries, timeline)

      ACTIVITY_START_ALL(entries, timeline, MPI_ALLREDUCE)
      int64_t num_elements = 0;
      for (auto& e : entries) {
        num_elements += e.tensor->shape().num_elements();
      }
      MPI_CHECK(entries, "MPI_Allreduce",
                MPI_Allreduce(MPI_IN_PLACE, (void*)buffer_data,
                              (int)num_elements,
                              GetMPIDataType(first_entry.tensor), MPI_SUM,
                              horovod_global.mpi_comm))
      ACTIVITY_END_ALL(entries, timeline)

      // Copy memory out of the fusion buffer.
      ACTIVITY_START_ALL(entries, timeline, MEMCPY_OUT_FUSION_BUFFER)
      offset = 0;
      for (auto& e : entries) {
        void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
#if HAVE_CUDA
        if (on_gpu) {
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(
                         (void*)e.output->data(), buffer_data_at_offset,
                         (size_t)e.tensor->size(), cudaMemcpyDeviceToDevice,
                         horovod_global.streams[first_entry.device]))
        } else {
#endif
          std::memcpy((void*)e.output->data(), buffer_data_at_offset,
                      (size_t)e.tensor->size());
#if HAVE_CUDA
        }
#endif
        offset += e.tensor->size();
      }
#if HAVE_CUDA
      if (on_gpu) {
        CUDA_CHECK(
            entries, "cudaStreamSynchronize",
            cudaStreamSynchronize(horovod_global.streams[first_entry.device]))
      }
#endif
      ACTIVITY_END_ALL(entries, timeline)
    } else {
      auto& e = first_entry;
      ACTIVITY_START_ALL(entries, timeline, MPI_ALLREDUCE)
      const void* sendbuf = e.tensor->data() == e.output->data()
                                ? MPI_IN_PLACE
                                : e.tensor->data();
      MPI_CHECK(entries, "MPI_Allreduce",
                MPI_Allreduce(sendbuf, (void*)e.output->data(),
                              (int)e.tensor->shape().num_elements(),
                              GetMPIDataType(e.tensor), MPI_SUM,
                              horovod_global.mpi_comm))
      ACTIVITY_END_ALL(entries, timeline)
    }

    for (auto& e : entries) {
      timeline.End(e.tensor_name, e.output);
      e.callback(Status::OK());
    }
  } else if (response.response_type() == MPIResponse::BROADCAST) {
    assert(entries.size() == 1);
    auto first_entry = entries[0];

    // On root rank, MPI_Bcast sends data, on other ranks it receives data.
    void* data_ptr;
    if (horovod_global.rank == first_entry.root_rank) {
      data_ptr = (void*)first_entry.tensor->data();
    } else {
      data_ptr = (void*)first_entry.output->data();
    }

    ACTIVITY_START_ALL(entries, timeline, MPI_BCAST)
    MPI_CHECK(entries, "MPI_Bcast",
              MPI_Bcast(data_ptr, (int)first_entry.tensor->shape().num_elements(),
                        GetMPIDataType(first_entry.tensor), first_entry.root_rank,
                        horovod_global.mpi_comm))
    ACTIVITY_END_ALL(entries, timeline)

    timeline.End(first_entry.tensor_name, first_entry.output);
    first_entry.callback(Status::OK());
  } else if (response.response_type() == MPIResponse::ERROR) {
    assert(entries.size() == 1);
    auto e = entries[0];

    status = Status::PreconditionError(response.error_message());
    timeline.End(e.tensor_name, nullptr);
    e.callback(status);
  }
}

// Report Tensors that were submitted to be reduced, gathered or broadcasted by
// some ranks but not others and are waiting for long time to get processed.
void CheckForStalledTensors(HorovodGlobalState& state) {
  bool preamble = false;
  auto now = std::chrono::steady_clock::now();
  for (auto& m : *state.message_table) {
    auto tensor_name = m.first;
    std::vector<MPIRequest>& messages = std::get<0>(m.second);
    std::chrono::steady_clock::time_point start_at = std::get<1>(m.second);

    if (now - start_at > STALL_WARNING_TIME) {
      if (!preamble) {
        std::cerr << "WARNING: One or more tensors were submitted to be "
                     "reduced, gathered or broadcasted by subset of ranks and "
                     "are waiting for remainder of ranks for more than "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         STALL_WARNING_TIME)
                         .count()
                  << " seconds. ";
        std::cerr << "This may indicate that different ranks are trying to "
                     "submit different tensors or that only subset of ranks is "
                     "submitting tensors, which will cause deadlock. " << std::endl;
        std::cerr << "Stalled ops:" << std::endl;
        preamble = true;
      }
      std::cerr << tensor_name;
      std::cerr << " [missing ranks:";
      std::unordered_set<int32_t> ready_ranks;
      bool missing_preamble = false;
      for (auto msg_iter = messages.begin(); msg_iter != messages.end();
           msg_iter++) {
             ready_ranks.insert(msg_iter->request_rank());
      }
      for (int32_t rank = 0; rank < state.size; rank++) {
        if (ready_ranks.find(rank) == ready_ranks.end()) {
          if (!missing_preamble) {
            std::cerr << " ";
            missing_preamble = true;
          } else {
            std::cerr << ", ";
          }
          std::cerr << rank;
        }
      }
      std::cerr << "]" << std::endl;
    }
  }
}

// The MPI background thread loop coordinates all the MPI processes and the
// tensor reductions. The design of the communicator mechanism is limited by a
// few considerations:
//
//      1. Some MPI implementations require all MPI calls to happen from a
//      single thread. Since TensorFlow may use several threads for graph
//      processing, this means we must have our own dedicated thread for dealing
//      with MPI.
//      2. We want to gracefully handle errors, when MPI processes do not
//      properly agree upon what should happen (such as mismatched types or
//      shapes). To do so requires the MPI processes to know about the shapes
//      and types of the relevant tensors on the other processes.
//      3. The MPI reductions and gathers should be able to happen in parallel
//      with other ongoing operations. This means that they cannot be blocking
//      ops, but rather must be async ops, the execution of which happens on a
//      separate thread.
//      4. We cannot guarantee that all the MPI processes reduce their tensors
//      in the same order, so we cannot dispatch one thread per tensor,
//      otherwise we may end up dispatching many blocked threads and never make
//      progress if we have a thread pool limit.
bool RunLoopOnce(HorovodGlobalState& state, bool is_coordinator);
bool RunTwoStageLoopOnce(HorovodGlobalState& state, bool is_coordinator,
    bool is_local_coordinator);
void BackgroundThreadLoop(HorovodGlobalState& state) {
  // Initialize MPI. This must happen on the background thread, since not all
  // MPI implementations support being called from multiple threads.
  //
  // In some cases MPI library has multi-threading support, but it slows down
  // certain components, e.g. OpenIB BTL in OpenMPI gets disabled if
  // MPI_THREAD_MULTIPLE is requested.
  //
  // By default, we will ask for multiple threads, so other libraries like
  // mpi4py can be used together with Horovod if multi-threaded MPI is
  // installed.
  auto mpi_threads_disable = std::getenv("HOROVOD_MPI_THREADS_DISABLE");
  int required = MPI_THREAD_MULTIPLE;
  if (mpi_threads_disable != nullptr &&
      std::strtol(mpi_threads_disable, nullptr, 10) > 0) {
    required = MPI_THREAD_FUNNELED;
  }
  int provided;
  MPI_Init_thread(NULL, NULL, required, &provided);

  // Create a private MPI communicator for Horovod to avoid collisions with
  // other threads using MPI.
  MPI_Comm mpi_comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);

  // Get MPI rank to determine if we are rank zero.
  int rank;
  MPI_Comm_rank(mpi_comm, &rank);
  bool is_coordinator = rank == 0;

  // Get MPI size to determine how many tensors to wait for before reducing.
  int size;
  MPI_Comm_size(mpi_comm, &size);

  // Determine local rank by querying the local communicator.
  MPI_Comm local_comm;
  MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &local_comm);
  int local_rank, local_size;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);
  std::vector<int> local_comm_ranks((size_t)local_size);
  local_comm_ranks[local_rank] = rank;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_comm_ranks.data(), 1,
                MPI_INT, local_comm);
  bool is_local_coordinator = local_rank == 0;

  // Determine if cluster is homogeneous, i.e., if every node has the same
  // local_size
  auto local_sizes = new int[size];
  MPI_Allgather(&local_size, 1, MPI_INT, local_sizes, 1, MPI_INT,
                mpi_comm);

  bool is_homogeneous = true;
  for (int i = 0; i < size; i++) {
    if (local_sizes[i] != local_size) {
      is_homogeneous = false;
      break;
    }
  }
  delete[] local_sizes;
  state.is_homogeneous = is_homogeneous;

  // Set up cross-communicator in case of hierarchical allreduce.
  MPI_Comm cross_comm;
  MPI_Comm_split(mpi_comm, local_rank, rank, &cross_comm);
  int cross_rank, cross_size;
  MPI_Comm_rank(cross_comm, &cross_rank);
  MPI_Comm_size(cross_comm, &cross_size);

  // Create custom MPI float16 data type.
  MPI_Datatype mpi_float16_t;
  MPI_Type_contiguous(2, MPI_BYTE, &mpi_float16_t);
  MPI_Type_commit(&mpi_float16_t);

  state.rank = rank;
  state.local_rank = local_rank;
  state.cross_rank = cross_rank;
  state.size = size;
  state.local_size = local_size;
  state.cross_size = cross_size;
  state.mpi_comm = mpi_comm;
  state.local_comm = local_comm;
  state.cross_comm = cross_comm;
  state.mpi_float16_t = mpi_float16_t;
  state.mpi_threads_supported = (provided == MPI_THREAD_MULTIPLE);
  state.local_comm_ranks = local_comm_ranks;

  // Open the timeline file on coordinator.
  auto horovod_timeline = std::getenv("HOROVOD_TIMELINE");
  if (is_coordinator && horovod_timeline != nullptr) {
    state.timeline.Initialize(std::string(horovod_timeline));
  }

  // Override Tensor Fusion threshold, if it's set.
  auto horovod_fusion_threshold = std::getenv("HOROVOD_FUSION_THRESHOLD");
  if (horovod_fusion_threshold != nullptr) {
    state.tensor_fusion_threshold =
        std::strtol(horovod_fusion_threshold, nullptr, 10);
  }

  // Override the cycle times and low-latency threshold.
  auto horovod_cycle_time = std::getenv("HOROVOD_CYCLE_TIME");
  if (horovod_cycle_time != nullptr) {
    state.cycle_time_ms = std::strtof(horovod_cycle_time, nullptr);
  }

  // Set flag for hierarchical allreduce. Ignore if Horovod is running on a
  // single node.
  auto horovod_hierarchical_allreduce =
      std::getenv("HOROVOD_HIERARCHICAL_ALLREDUCE");
  if (horovod_hierarchical_allreduce != nullptr &&
      std::strtol(horovod_hierarchical_allreduce, nullptr, 10) > 0 &&
      cross_size > 1) {
    state.hierarchical_allreduce = true;
  }

  // Set flag for two level communication strategy
  auto horovod_two_stage_loop =
      std::getenv("HOROVOD_TWO_STAGE_LOOP");
  if (horovod_two_stage_loop != nullptr && std::atoi(horovod_two_stage_loop) != 0) {
    state.two_stage_loop = true;
  }

  auto horovod_allreduce_mode =
      std::getenv("HOROVOD_ALLREDUCE_MODE");
  if (horovod_allreduce_mode != nullptr && std::atoi(horovod_allreduce_mode) != 0) {
    state.allreduce_mode = std::atoi(horovod_allreduce_mode);
    if (state.allreduce_mode != 1) {
      if (state.rank == RANK_ZERO) {
        std::cerr << "HOROVOD_ALLREDUCE_MODE = " << state.allreduce_mode << " not valid.";
        std::cerr << "Reverting to default (HOROVOD_ALLREDUCE_MODE = 0)." << std::endl;
      }
      state.allreduce_mode = 0;
    } else if (state.hierarchical_allreduce && state.allreduce_mode != 0) {
      if (state.rank == RANK_ZERO) {
        std::cerr << "HOROVOD_ALLREDUCE_MODE = " << state.allreduce_mode << " and ";
        std::cerr << "HOROVOD_HIERARCHICAL_ALLREDUCE are incompatible options.";
        std::cerr << "Reverting to default (HOROVOD_ALLREDUCE_MODE = 0)." << std::endl;
      }
      state.allreduce_mode = 0;
    } else if (!state.is_homogeneous && state.allreduce_mode != 0) {
      if (state.rank == RANK_ZERO) {
        std::cerr << "HOROVOD_ALLREDUCE_MODE = " << state.allreduce_mode << " is ";
        std::cerr << "not supported on heterogenous configurations across nodes.";
        std::cerr << "Reverting to default (HOROVOD_ALLREDUCE_MODE = 0)." << std::endl;
      }
      state.allreduce_mode = 0;
    }
  }

  auto horovod_fixed_payload =
      std::getenv("HOROVOD_FIXED_PAYLOAD");
  if (horovod_fixed_payload != nullptr && std::atoi(horovod_fixed_payload) != 0) {
    state.fixed_payload = std::atoi(horovod_fixed_payload);
  }

  // add new environment variables
  auto horovod_multiple_transfer_mode =
      std::getenv("HOROVOD_MULTIPLE_TRANSFER_MODE");
  if (horovod_multiple_transfer_mode != nullptr && std::atoi(horovod_multiple_transfer_mode) != 0) {
    state.multiple_transfer_mode = true;
  }

  auto horovod_cubic_transfer_mode =
      std::getenv("HOROVOD_CUBIC_TRANSFER_MODE");
  if (horovod_cubic_transfer_mode != nullptr && std::atoi(horovod_cubic_transfer_mode) != 0) {
    state.cubic_transfer_mode = true;
  }

  auto horovod_quadratic_transfer_mode =
      std::getenv("HOROVOD_QUADRATIC_TRANSFER_MODE");
  if (horovod_quadratic_transfer_mode != nullptr && std::atoi(horovod_quadratic_transfer_mode) != 0) {
    state.quadratic_transfer_mode = true;
  }

  auto horovod_fixed_transfersize =
      std::getenv("HOROVOD_FIXED_TRANSFERSIZE");
  if (horovod_fixed_transfersize != nullptr && std::atoi(horovod_fixed_transfersize) != 0) {
    state.fixed_transfersize = std::atoi(horovod_fixed_transfersize);
  }

  auto horovod_triple_streams4allreduce =
      std::getenv("HOROVOD_TRIPLE_STREAMS4ALLREDUCE");
  if (horovod_triple_streams4allreduce != nullptr && std::atoi(horovod_triple_streams4allreduce) != 0) {
    state.triple_streams4allreduce = true;
  }

  auto horovod_division_positions =
      std::getenv("HOROVOD_DIVISION_POSITIONS");
  if (horovod_division_positions != nullptr ) {
    char del = ',';
    state.division_positions = conv_text2numbers( horovod_division_positions, del );

  }

  // Initialize the tensor count table. No tensors are available yet.
  if (is_coordinator) {
    state.message_table = std::unique_ptr<MessageTable>(new MessageTable());
  }
  if (is_local_coordinator && state.two_stage_loop) {
    state.local_message_table = std::unique_ptr<MessageTable>(new MessageTable());
  }

  if (state.fixed_payload != 0) {
    state.fixed_message_table = std::unique_ptr<MessageTable>(new MessageTable());
  }

#if HOROVOD_GPU_ALLREDUCE=='N'
  cudaSetDevice( state.local_rank );
  std::vector<int32_t> gpu_device_id( state.local_rank );

  // Ensure NCCL communicator is in the map before executing reduction.
  ncclComm_t& nccl_comm = state.nccl_comms[ gpu_device_id ];
  ncclComm_t& nccl_local_comm = state.nccl_local_comms[ gpu_device_id ];
  ncclComm_t& nccl_cross_comm = state.nccl_cross_comms[ gpu_device_id ];
  
  // Initialize global NCCL communicator
  if (state.allreduce_mode == 0 && nccl_comm == nullptr) {
  
    int nccl_rank, nccl_size;
    MPI_Comm nccl_id_bcast_comm;
    if (state.hierarchical_allreduce) {
      nccl_rank = state.local_rank;
      nccl_size = state.local_size;
      nccl_id_bcast_comm = state.local_comm;
    } else {
      nccl_rank = state.rank;
      nccl_size = state.size;
      nccl_id_bcast_comm = state.mpi_comm;
    }
  
    ncclUniqueId nccl_id;

    if (state.rank == 0) {
      ncclGetUniqueId(&nccl_id);
    }
  
    MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, nccl_id_bcast_comm);
  
    ncclComm_t new_nccl_comm;
    ncclCommInitRank(&new_nccl_comm, nccl_size, nccl_id, nccl_rank);
    nccl_comm = new_nccl_comm;
  
    // Barrier helps NCCL to synchronize after initialization and avoid
    // deadlock that we've been seeing without it.
    MPI_Barrier( state.mpi_comm);
  
  }
  
  // Iniitialize local and cross NCCL communicators
  if ( state.allreduce_mode == 1 && nccl_local_comm == nullptr &&
             nccl_cross_comm == nullptr ) {
  
    ncclUniqueId nccl_id;
    if (state.local_rank == 0) {
      ncclGetUniqueId(&nccl_id);
    }
  
    MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, state.local_comm);
  
    ncclComm_t new_nccl_local_comm;
    ncclCommInitRank(&new_nccl_local_comm, state.local_size, nccl_id, state.local_rank);
  
    nccl_local_comm = new_nccl_local_comm;
  
    MPI_Barrier( state.local_comm);
  
    if ( state.rank < state.local_size) {
      ncclGetUniqueId(&nccl_id);
    }
  
    MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, state.cross_comm);
  
    ncclComm_t new_nccl_cross_comm;
    ncclCommInitRank(&new_nccl_cross_comm, state.cross_size, nccl_id, state.cross_rank);
    nccl_cross_comm = new_nccl_cross_comm;
  
    MPI_Barrier(state.cross_comm);
  
  }
#endif

  auto horovod_prepare = std::getenv("HOROVOD_PREPARE");
  if (horovod_prepare != nullptr && std::atoi(horovod_prepare) != 0) {

    auto hvd_cpu_context = CreateMXOpContext(CPU_DEVICE_ID); 
    auto& cpu_buffer = horovod_global.tensor_fusion_buffers[std::make_tuple(CPU_DEVICE_ID, hvd_cpu_context->framework())];
    
    if (cpu_buffer == nullptr) {
      Status status = AllocateBuffer(hvd_cpu_context, cpu_buffer);
      if (!status.ok()) {
    	return;
      }
    }

    int device = state.local_rank;
    auto hvd_gpu_context = CreateMXOpContext(device);
    auto& gpu_buffer = horovod_global.tensor_fusion_buffers[std::make_tuple(device, hvd_gpu_context->framework())];
    
    if (gpu_buffer == nullptr) {
      Status status = AllocateBuffer(hvd_gpu_context, gpu_buffer);
      if (!status.ok()) {
    	return;
      }
    }
    
  }

  MPI_Barrier( state.mpi_comm );

  // Signal that initialization is completed.
  state.initialization_done = true;

  // Iterate until shutdown.
  if (!state.two_stage_loop) {
    while (RunLoopOnce(state, is_coordinator)) {};
  } else {
    while (RunTwoStageLoopOnce(state, is_coordinator, is_local_coordinator)) {};
  }

  // TODO: init.cu:645 WARN Cuda failure 'driver shutting down'
  //#if HAVE_NCCL
  //  for (auto it = horovod_global.streams.begin();
  //       it != horovod_global.streams.end(); it++) {
  //    cudaStreamSynchronize(it->second);
  //  }
  //  for (auto it = horovod_global.nccl_comms.begin();
  //       it != horovod_global.nccl_comms.end(); it++) {
  //    ncclCommDestroy(it->second);
  //  }
  //#endif

  // Notify all outstanding operations that Horovod has been shut down
  // and clear up the tensor table and message queue.
  std::vector<StatusCallback> callbacks;
  {
    std::lock_guard<std::mutex> guard(state.mutex);
    for (auto& e : state.tensor_table) {
      callbacks.emplace_back(e.second.callback);
    }
    state.tensor_table.clear();
    while (!state.message_queue.empty()) {
      state.message_queue.pop();
    }
  }
  for (auto& cb : callbacks) {
    cb(SHUT_DOWN_ERROR);
  }

  // Free batched memcpy pointers
  state.pack_ptrs.free();

  MPI_Comm_free(&state.mpi_comm);
  MPI_Comm_free(&state.local_comm);
  MPI_Comm_free(&state.cross_comm);
  MPI_Type_free(&state.mpi_float16_t);

#if HAVE_DDL
  // ddl_finalize calls MPI_Finalize
  ddl_finalize();
#else
  MPI_Finalize();
#endif
}

// In fixed payload case, all ranks can execute logic independently. This function
// encapsulates that logic.
void RunBypass(std::queue<MPIRequest>& message_queue, HorovodGlobalState& state) {
    // Using set to get consistently ordered list
    std::set<std::string> ready_to_reduce_fixed;

    while (!message_queue.empty()) {
       // Pop the first available message
       MPIRequest message = message_queue.front();
       message_queue.pop();

       IncrementTensorCount(state.fixed_message_table, message, 1);
       ready_to_reduce_fixed.insert(message.tensor_name());
    }

    // Every rank forms own response
    std::map<MPIDataType, std::deque<MPIResponse>> responses_by_type;
    MPIResponseList response_list;
    {
    
      std::lock_guard<std::mutex> guard(state.mutex);
      for (auto& tensor_name : ready_to_reduce_fixed) {
        MPIResponse response =
            ConstructMPIResponse(state.fixed_message_table, tensor_name);
        auto& entry = state.tensor_table[response.tensor_names()[0]];
        responses_by_type[entry.tensor->dtype()].push_back(std::move(response));
      }
  
      PopulateMPIResponseList(response_list, responses_by_type, state);
    }

    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    for (auto& response : response_list.responses()) {
      PerformOperation(state.tensor_table, response);
    }
}


// The coordinator currently follows a master-worker paradigm. Rank zero acts
// as the master (the "coordinator"), whereas all other ranks are simply
// workers. Each rank runs its own background thread which progresses in ticks.
// In each tick, the following actions happen:
//
//      a) The workers send an MPIRequest to the coordinator, indicating what
//      they would like to do (which tensor they would like to gather and
//      reduce, as well as their shape and type). They repeat this for every
//      tensor that they would like to operate on.
//
//      b) The workers send an empty "DONE" message to the coordinator to
//      indicate that there are no more tensors they wish to operate on.
//
//      c) The coordinator receives the MPIRequests from the workers, as well
//      as from its own TensorFlow ops, and stores them in a request table. The
//      coordinator continues to receive MPIRequest messages until it has
//      received MPI_SIZE number of empty "DONE" messages.
//
//      d) The coordinator finds all tensors that are ready to be reduced,
//      gathered, or all operations that result in an error. For each of those,
//      it sends an MPIResponse to all the workers. When no more MPIResponses
//      are available, it sends a "DONE" response to the workers. If the process
//      is being shutdown, it instead sends a "SHUTDOWN" response.
//
//      e) The workers listen for MPIResponse messages, processing each one by
//      doing the required reduce or gather, until they receive a "DONE"
//      response from the coordinator. At that point, the tick ends.
//      If instead of "DONE" they receive "SHUTDOWN", they exit their background
//      loop.
bool RunLoopOnce(HorovodGlobalState& state, bool is_coordinator) {
  // The coordinator sends a SHUTDOWN message to trigger shutdown.
  bool should_shut_down = false;

  // This delay determines thread frequency and MPI message latency
  auto sleep_duration =
      state.last_cycle_start +
      std::chrono::microseconds(long(state.cycle_time_ms * 1000.)) -
      std::chrono::steady_clock::now();
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }

  // Use barrier to sync Horovod worker thread timings
  if( !state.multiple_transfer_mode ){
    MPI_Barrier(state.mpi_comm);
  }
  state.last_cycle_start = std::chrono::steady_clock::now();

  // Copy the data structures from global state under this lock.
  // However, don't keep the lock for the rest of the loop, so that
  // enqueued stream callbacks can continue.

  // add one status which represents Allreduce readiness of each process
  int status[4]; // int status[3];

  std::queue<MPIRequest> message_queue;
  {
    std::lock_guard<std::mutex> guard(state.mutex);

    if (state.fixed_payload == 0) {
      while (!state.message_queue.empty()) {
        MPIRequest message = state.message_queue.front();
        state.message_queue.pop();
        message_queue.push(message);
      }
    } else if ( state.multiple_transfer_mode ) {
      // Check to see if all ranks have tensors ready to go for fixed transfer case.
      // If MPI operations are not Allreduce, original horovod code is executed.

      status[3] = -1;
      int nmessages = state.message_queue.size();
      if( !state.initial_Bcast_done ) { // Bcast phase for weight initialization of tensors
        if( nmessages >= state.fixed_payload ){
          // check all Bcast requests are stored in queue
          MPIRequest message;
          int num_bcast_request = 0;
          for (int i = 0; i < nmessages; i++) {
            message = state.message_queue.front();
            state.message_queue.pop();
            if (message.request_type() != MPIRequest::RequestType::ALLREDUCE) {
              status[3] = 1;
              num_bcast_request++;
              message_queue.push(message);
            }
          }

          if( num_bcast_request < state.fixed_payload ) {
            // Bcast operations are executed after all MPIRequests are stored in a queue.
            status[3] = -1;
            nmessages = message_queue.size();
            for (int i = 0; i < nmessages; i++) {
              message = message_queue.front();
              message_queue.pop();
              state.message_queue.push(message);
            }
          } else {
            state.initial_Bcast_done = true;
          }
        }
      } else { // Allreduce phase

        int count, begin_point, end_point;
        std::string substr;
        int tensor_id = 0;
        MPIRequest message;
        int offset = state.layer_offset;
        int fixedsize = state.fixed_transfersize;

        count = state.fp32_layer_count;

        if( !state.division_positions.empty() ){
          begin_point = state.division_positions[offset + 1];
          end_point = state.division_positions[offset];
        } else if( horovod_global.cubic_transfer_mode ){
          begin_point = state.fixed_payload - ( offset + 1) * ( offset + 1) * ( offset + 1) * fixedsize;
          end_point = state.fixed_payload - offset * offset * offset * fixedsize;
          begin_point = ( begin_point > 0 ) ? begin_point : 0;
          end_point = ( end_point > 0 ) ? end_point : 0;
        } else if( horovod_global.quadratic_transfer_mode ) {
          begin_point = state.fixed_payload - ( offset +  1) * (offset + 1) * fixedsize;
          end_point = state.fixed_payload - offset * offset * fixedsize;
          begin_point = ( begin_point > 0 ) ? begin_point : 0;
          end_point = ( end_point > 0 ) ? end_point : 0;
        } else {
          begin_point = state.fixed_payload - (offset + 1) * fixedsize;
          end_point = begin_point + fixedsize;
        }

        // Count MPI requests which have tensor_id between begin_point and end_point.
        for (int i = 0; i < nmessages; i++) {
          message = state.message_queue.front();
          state.message_queue.pop();
          substr = message.tensor_name().substr(10);
          tensor_id = atoi( substr.c_str() );
          if( (tensor_id >= begin_point ) && ( tensor_id < end_point ) ){ // Check tensor_id
            if( message.tensor_type() == HOROVOD_FLOAT16 ){ // Check data type
              message_queue.push(message);
            } else { // If tensor type is fp32, the tensor is enqueued in fp32_message_queue.
              state.fp32_message_queue.push( message );
              state.fp32_layer_count++;
            }
            count++;
          } else {
            state.message_queue.push(message);
          }
        }

        if( count == ( end_point - begin_point ) ){
          status[3] = 0;
          state.fp32_layer_count = 0;
        } else {
          while( !message_queue.empty()){
            message = message_queue.front();
            message_queue.pop();
            state.message_queue.push(message);
          }
        }
      }

    } else {
      // Check to see if all ranks have all tensors ready to go for fixed payload case.
      // Saves wasted loop cycles with only partially prepared ranks.
      int nmessages = state.message_queue.size();
      for (int i = 0; i < nmessages; i++) {
        MPIRequest message = state.message_queue.front();
        state.message_queue.pop();
        // Forward any messages that aren't allreduce
        if (message.request_type() != MPIRequest::RequestType::ALLREDUCE) {
          message_queue.push(message);
        // Replace allreduce messages to state message queue
        } else {
          state.message_queue.push(message);
        }
      }

      status[0] = (int) ((int) state.message_queue.size() == state.fixed_payload &&
                         message_queue.size() == 0); // sum of ranks with fixed_payload tensors ready
      status[1] = (int) message_queue.size() > 0; // sum of ranks with non allreduce messages
      status[2] = (int) state.shut_down; // sum of ranks requesting shutdown

      MPI_Allreduce(MPI_IN_PLACE, status, 3, MPI_INT, MPI_SUM, state.mpi_comm);

      if (status[0] == state.size) {
        // There should only be allreduce messages in the state queue at this point
        while (!state.message_queue.empty()) {
          MPIRequest message = state.message_queue.front();
          state.message_queue.pop();
          message_queue.push(message);
        }
      }
    }
  }

  // If fixed payload or multiple transfer modes, Allreduce operations is executed through RunBypass
  if ( state.multiple_transfer_mode ){ 
    if( status[3] == 0 ){
      RunBypass(message_queue, state);

      state.layer_offset++;
      int stat_flag = 0;
      int count;
      int offset = state.layer_offset;
      int fixedsize = state.fixed_transfersize;

      if( !state.division_positions.empty() ){
        stat_flag = ( state.division_positions.size() == (unsigned int)(offset + 1) ) ? 1 : 0;
      }else{
        if( state.cubic_transfer_mode ){
          count = std::min( offset * offset * offset * fixedsize, state.fixed_payload);
        }else if( state.quadratic_transfer_mode ){
          count = std::min( offset * offset * fixedsize, state.fixed_payload);
        }else{
          count = offset * fixedsize;
        }
        stat_flag = ( count == state.fixed_payload ) ? 1 : 0;
      }

      if( stat_flag ){
        RunBypass(state.fp32_message_queue, state);
        state.layer_offset = 0;
      }

      return 1; 
    } else if( status[3] == -1 ){
      if( state.shut_down ) should_shut_down = true;
      return !should_shut_down;
    }
  } else {
    // Check if we can use fast path which bypasses global coordination
    if (state.fixed_payload != 0 && status[0] == state.size) {
      should_shut_down = (status[2] > 0);
      RunBypass(message_queue, state);
      return !should_shut_down;
    } else if (state.fixed_payload != 0 && status[1] == 0) {
      // Quick return if there are no messages
      should_shut_down = (status[2] > 0);
      return !should_shut_down;
    }
  }

  // Collect all tensors that are ready to be reduced. Record them in the
  // tensor count table (rank zero) or send them to rank zero to be
  // recorded (everyone else).
  std::vector<std::string> ready_to_reduce;
  if (is_coordinator) {
    while (!message_queue.empty()) {
      // Pop the first available message message
      MPIRequest message = message_queue.front();
      message_queue.pop();

      bool reduce =
          IncrementTensorCount(state.message_table, message, state.size);
      if (reduce) {
        ready_to_reduce.push_back(message.tensor_name());
      }
    }

    // Rank zero has put all its own tensors in the tensor count table.
    // Now, it should count all the tensors that are coming from other
    // ranks at this tick.


    // Note: Changing to MPI_Gather from original MPI_Gatherv
    // 1. Get maximum message length across ranks.
    int max_length;
    int encoded_message_length = 0;
    MPI_Allreduce(&encoded_message_length, &max_length, 1, MPI_INT, MPI_MAX, state.mpi_comm);

    // 3. Collect messages from every rank.
    auto buffer = new char[state.size*max_length];
    MPI_Gather(MPI_IN_PLACE, max_length, MPI_BYTE, buffer, max_length, MPI_BYTE, RANK_ZERO, state.mpi_comm);

    // 4. Process messages.
    for (int i = 1; i < state.size; i++) {
      std::string received_data;
      received_data = std::string(buffer + i*max_length, (size_t)max_length);

      MPIRequestList received_message_list;
      MPIRequestList::ParseFromString(received_message_list, received_data);
      for (auto& received_message : received_message_list.requests()) {
        auto& received_name = received_message.tensor_name();

        bool reduce = IncrementTensorCount(state.message_table,
                                           received_message, state.size);
        if (reduce) {
          ready_to_reduce.push_back(received_name);
        }
      }
      if (received_message_list.shutdown()) {
        // Received SHUTDOWN request from one of the workers.
        state.shut_down = true;
      }
    }

    // 5. Free buffers.
    delete[] buffer;

    // At this point, rank zero should have a fully updated tensor count
    // table and should know all the tensors that need to be reduced or
    // gathered, and everyone else should have sent all their information
    // to rank zero. We can now do reductions and gathers; rank zero will
    // choose which ones and in what order, and will notify the other ranks
    // before doing each reduction.

    // Mixed-precision training may produce tensors with FP32 and FP16 precision
    // in a mixed ordering. To enable more tensor fusion, process reponses
    // by type.
    std::map<MPIDataType, std::deque<MPIResponse>> responses_by_type;

    for (auto& tensor_name : ready_to_reduce) {
      MPIResponse response =
          ConstructMPIResponse(state.message_table, tensor_name);
      auto& entry = state.tensor_table[response.tensor_names()[0]];
      responses_by_type[entry.tensor->dtype()].push_back(std::move(response));
    }

    MPIResponseList response_list;
    response_list.set_shutdown(state.shut_down);
    should_shut_down = state.shut_down;

    PopulateMPIResponseList(response_list, responses_by_type, state);

    // Notify all nodes which tensors we'd like to reduce at this step.
    std::string encoded_response;
    MPIResponseList::SerializeToString(response_list, encoded_response);
    int encoded_response_length = (int)encoded_response.length() + 1;
    MPI_Bcast(&encoded_response_length, 1, MPI_INT, RANK_ZERO, state.mpi_comm);
    MPI_Bcast((void*)encoded_response.c_str(), encoded_response_length,
              MPI_BYTE, RANK_ZERO, state.mpi_comm);

    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    for (auto& response : response_list.responses()) {
      PerformOperation(state.tensor_table, response);
    }

    // Check for stalled tensors.
    if (std::chrono::steady_clock::now() - state.last_stall_check >
        STALL_WARNING_TIME) {
      CheckForStalledTensors(state);
      state.last_stall_check = std::chrono::steady_clock::now();
    }
  } else {
    std::string encoded_message;
    MPIRequestList message_list;
    message_list.set_shutdown(state.shut_down);
    while (!message_queue.empty()) {
      message_list.add_requests(message_queue.front());
      message_queue.pop();
    }
    MPIRequestList::SerializeToString(message_list, encoded_message);
    int encoded_message_length = (int)encoded_message.length() + 1;
    int max_length;
    MPI_Allreduce(&encoded_message_length, &max_length, 1, MPI_INT, MPI_MAX, state.mpi_comm);
    encoded_message.resize(max_length-1);
    MPI_Gather((void*)encoded_message.c_str(), max_length, MPI_BYTE, nullptr, 0, MPI_BYTE, RANK_ZERO, state.mpi_comm);

    int msg_length;
    MPI_Bcast(&msg_length, 1, MPI_INT, RANK_ZERO, state.mpi_comm);
    auto buffer = new char[msg_length];
    MPI_Bcast(buffer, msg_length, MPI_BYTE, RANK_ZERO, state.mpi_comm);
    std::string received_message(buffer, (size_t)msg_length);
    MPIResponseList response_list;
    MPIResponseList::ParseFromString(response_list, received_message);
    delete[] buffer;

    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    for (auto& response : response_list.responses()) {
      PerformOperation(state.tensor_table, response);
    }

    if (response_list.shutdown()) {
      should_shut_down = true;
    }
  }

  return !should_shut_down;
}

bool RunTwoStageLoopOnce(HorovodGlobalState& state, bool is_coordinator,
    bool is_local_coordinator) {
  // The coordinator sends a SHUTDOWN message to trigger shutdown.
  bool should_shut_down = false;

  // This delay determines thread frequency and MPI message latency
  auto sleep_duration =
      state.last_cycle_start +
      std::chrono::microseconds(long(state.cycle_time_ms * 1000.)) -
      std::chrono::steady_clock::now();
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }

  // Use barrier to sync Horovod worker thread timings
  if( state.multiple_transfer_mode == 0 ) {
    MPI_Barrier(state.mpi_comm);
  }
  state.last_cycle_start = std::chrono::steady_clock::now();

  // Copy the data structures from global state under this lock.
  // However, don't keep the lock for the rest of the loop, so that
  // enqueued stream callbacks can continue.

  int status[4]; // extended status array. original is "int status[3]"
  std::queue<MPIRequest> message_queue;
  {

    std::lock_guard<std::mutex> guard(state.mutex);
    int nmessages;
    if (state.fixed_payload == 0) {
      while (!state.message_queue.empty()) {
        MPIRequest message = state.message_queue.front();
        state.message_queue.pop();
        message_queue.push(message);
      }

    } else if( state.multiple_transfer_mode ) {
      // Check to see if all ranks have tensors ready to go for fixed transfer case.
      // If stored MPI requests are Bcast, all processes wait for all requests are enqueueed and original horovod code is executed.

      status[3] = -1;
      nmessages = state.message_queue.size();
      if( !state.initial_Bcast_done ) { // Bcast phase for weight initialization of tensors
        // Check all Bcast requests are stored in queue
        if( nmessages >= state.fixed_payload ){
          MPIRequest message;
          int num_bcast_request = 0;
          // Count the number of Bcast requests
          for (int i = 0; i < nmessages; i++) {
            message = state.message_queue.front();
            state.message_queue.pop();
            if (message.request_type() != MPIRequest::RequestType::ALLREDUCE) {
              status[3] = 1;
              num_bcast_request++;
              message_queue.push(message);
            }
          }

          // If the number of Bcast requests is equal to that of tensors, Bcast operations are executed.
          if( num_bcast_request < state.fixed_payload ) {
            status[3] = -1;
            nmessages = message_queue.size();
            for (int i = 0; i < nmessages; i++) {
              message = message_queue.front();
              message_queue.pop();
              state.message_queue.push(message);
            }
          } else {
            state.initial_Bcast_done = true;
          }
        }
      } else { // Allreduce phase
        int count, begin_point, end_point;
        std::string substr;
        int tensor_id = 0;
        MPIRequest message;
        int offset = state.layer_offset;
        int fixedsize = state.fixed_transfersize;

        count = state.fp32_layer_count;

        if( !state.division_positions.empty() ){
          begin_point = state.division_positions[offset + 1];
          end_point = state.division_positions[offset];
        } else if( state.cubic_transfer_mode ){
          begin_point = state.fixed_payload - ( offset + 1) * ( offset + 1) * ( offset + 1) * fixedsize;
          end_point = state.fixed_payload - offset * offset * offset * fixedsize;
          begin_point = ( begin_point > 0 ) ? begin_point : 0;
          end_point = ( end_point > 0 ) ? end_point : 0;
        } else if( horovod_global.quadratic_transfer_mode ) {
          begin_point = state.fixed_payload - ( offset +  1) * (offset + 1) * fixedsize;
          end_point = state.fixed_payload - offset * offset * fixedsize;
          begin_point = ( begin_point > 0 ) ? begin_point : 0;
          end_point = ( end_point > 0 ) ? end_point : 0;
        } else {
          begin_point = state.fixed_payload - (offset + 1) * fixedsize;
          end_point = begin_point + fixedsize;
        }


        // Count the number of MPI requests which have tensor_id between begin_point and end_point.
        for (int i = 0; i < nmessages; i++) {
          message = state.message_queue.front();
          state.message_queue.pop();
          substr = message.tensor_name().substr(10);
          tensor_id = atoi( substr.c_str() );
          if( (tensor_id >= begin_point ) && ( tensor_id < end_point ) ){ // Check tensor_id
            if( message.tensor_type() == HOROVOD_FLOAT16 ){ // Check data type
              message_queue.push(message);
            } else { // If tensor type is fp32, the tensor is enqueued in fp32_message_queue.
              state.fp32_message_queue.push( message );
              state.fp32_layer_count++;
            }
            count++;
          } else {
            state.message_queue.push(message);
          }
        }

        if( count == ( end_point - begin_point ) ){
          status[3] = 0;
          state.fp32_layer_count = 0;
        } else {
          while( !message_queue.empty()){
            message = message_queue.front();
            message_queue.pop();
            state.message_queue.push(message);
          }
        }
      }
    } else {
      // Check to see if all ranks have all tensors ready to go for fixed payload case.
      // Saves wasted loop cycles with only partially prepared ranks.

      nmessages = state.message_queue.size();
      for (int i = 0; i < nmessages; i++) {
        MPIRequest message = state.message_queue.front();
        state.message_queue.pop();
        // Forward any messages that aren't allreduce
        if (message.request_type() != MPIRequest::RequestType::ALLREDUCE) {
          message_queue.push(message);
        // Replace allreduce messages to state message queue
        } else {
          state.message_queue.push(message);
        }
      }

      status[0] = (int) ((int) state.message_queue.size() == state.fixed_payload &&
                         message_queue.size() == 0); // sum of ranks with fixed_payload tensors ready
      status[1] = (int) message_queue.size() > 0; // sum of ranks with non allreduce messages
      status[2] = (int) state.shut_down; // sum of ranks requesting shutdown

      MPI_Allreduce(MPI_IN_PLACE, status, 3, MPI_INT, MPI_SUM, state.mpi_comm);

      if (status[0] == state.size) {
        // There should only be allreduce messages in the state queue at this point
        while (!state.message_queue.empty()) {
          MPIRequest message = state.message_queue.front();
          state.message_queue.pop();
          message_queue.push(message);
        }
      }
    }
  }

  // Check if we can use fast path which bypasses global coordination
  if ( state.multiple_transfer_mode ){
    if( status[3] == 0 ){

      RunBypass(message_queue, state);

      state.layer_offset++;
      int stat_flag = 0;
      int count;
      int offset = state.layer_offset;
      int fixedsize = state.fixed_transfersize;

      if( !state.division_positions.empty() ){
        stat_flag = ( state.division_positions.size() == (unsigned int)(offset + 1) ) ? 1 : 0;
      }else{
        if( state.cubic_transfer_mode ){
          count = std::min( offset * offset * offset * fixedsize, state.fixed_payload);
        }else if( state.quadratic_transfer_mode ){
          count = std::min( offset * offset * fixedsize, state.fixed_payload);
        }else{
          count = offset * fixedsize;
        }
        stat_flag = ( count == state.fixed_payload ) ? 1 : 0;
      }

      if( stat_flag ){
        RunBypass(state.fp32_message_queue, state);
        state.layer_offset = 0;
      }

      return 1;
    } else if( status[3] == -1 ){
      if( state.shut_down ) should_shut_down = true;
      return !should_shut_down;
    }
  } else {
    if (state.fixed_payload != 0 && status[0] == state.size) {
      should_shut_down = (status[2] > 0);
      RunBypass(message_queue, state);
      return !should_shut_down;
    } else if (state.fixed_payload != 0 && status[1] == 0) {
      // Quick return if there are no messages
      should_shut_down = (status[2] > 0);
      return !should_shut_down;
    }
  }

  std::vector<std::string> ready_to_reduce;
  if (is_coordinator || is_local_coordinator) {

    std::string encoded_message;
    MPIRequestList message_list;
    message_list.set_shutdown(state.shut_down);
    while (!message_queue.empty()) {
      message_list.add_requests(message_queue.front());
      message_queue.pop();
    }

    MPIRequestList::SerializeToString(message_list, encoded_message);
    int encoded_message_length = (int)encoded_message.length() + 1;
    int max_length;

    // Local coordinators processes local node requests
    MPI_Allreduce(&encoded_message_length, &max_length, 1, MPI_INT, MPI_MAX, state.local_comm);

    encoded_message.resize(max_length-1);

    auto local_buffer = new char[state.local_size*max_length];

    MPI_Gather((void*)encoded_message.c_str(), max_length, MPI_BYTE, local_buffer, max_length, MPI_BYTE, RANK_ZERO, state.local_comm);

    MPIRequestList local_message_list;
    for (int i = 0; i < state.local_size; i++) {
      std::string received_data;
      received_data = std::string(local_buffer + i*max_length, (size_t)max_length);

      MPIRequestList received_message_list;
      MPIRequestList::ParseFromString(received_message_list, received_data);
      for (auto& received_message : received_message_list.requests()) {

        bool reduce = IncrementTensorCount(state.local_message_table,
                                           received_message, state.local_size);
        if (reduce) {
          local_message_list.add_requests(received_message);
        }
      }
      if (received_message_list.shutdown()) {
        // Received SHUTDOWN request from one of the local workers.
         local_message_list.set_shutdown(true);
      }
    }
    delete[] local_buffer;

    // Local coordinators send requests to global coordinator.
    std::string local_encoded_message;
    MPIRequestList::SerializeToString(local_message_list, local_encoded_message);
    int local_encoded_message_length = (int)local_encoded_message.length() + 1;

    MPI_Allreduce(&local_encoded_message_length, &max_length, 1, MPI_INT, MPI_MAX, state.cross_comm);
    local_encoded_message.resize(max_length-1);

    // Global coordinator processes requests from local coordinators.
    std::string received_data;
    if (is_coordinator) {
      auto buffer = new char[state.cross_size*max_length];
      MPI_Gather((void*)local_encoded_message.c_str(), max_length, MPI_BYTE, buffer, max_length, MPI_BYTE, RANK_ZERO, state.cross_comm);

      MPIRequestList global_message_list;
      for (int i = 0; i < state.cross_size; i++) {
        std::string received_data;
        received_data = std::string(buffer + i*max_length, (size_t)max_length);

        MPIRequestList received_message_list;
        MPIRequestList::ParseFromString(received_message_list, received_data);
        for (auto& received_message : received_message_list.requests()) {
          auto& received_name = received_message.tensor_name();

          bool reduce = IncrementTensorCount(state.message_table,
                                             received_message, state.cross_size);
          if (reduce) {
            global_message_list.add_requests(received_message);
            // need to erase here since global coordinator doesn't create MPI_responses
            state.message_table->erase(received_name);
          }
        }
        if (received_message_list.shutdown()) {
          // Received SHUTDOWN request from one of the local coordinators
          global_message_list.set_shutdown(true);
        }
      }

      delete[] buffer;

      std::string global_encoded_message;
      MPIRequestList::SerializeToString(global_message_list, global_encoded_message);
      int global_encoded_message_length = (int)global_encoded_message.length() + 1;

      MPI_Bcast(&global_encoded_message_length, 1, MPI_INT, RANK_ZERO, state.cross_comm);
      MPI_Bcast((void*)global_encoded_message.c_str(), global_encoded_message_length,
                MPI_BYTE, RANK_ZERO, state.cross_comm);

      received_data = global_encoded_message;

    } else {
      int global_encoded_message_length;
      MPI_Gather((void*)local_encoded_message.c_str(), max_length, MPI_BYTE, nullptr,  max_length, MPI_BYTE, RANK_ZERO, state.cross_comm);
      MPI_Bcast(&global_encoded_message_length, 1, MPI_INT, RANK_ZERO, state.cross_comm);

      auto buffer2 = new char[global_encoded_message_length];
      MPI_Bcast(buffer2, global_encoded_message_length,
                MPI_BYTE, RANK_ZERO, state.cross_comm);
      received_data = std::string(buffer2, (size_t)global_encoded_message_length);
      delete[] buffer2;
    }

    MPIRequestList received_message_list;
    MPIRequestList::ParseFromString(received_message_list, received_data);
    for (auto& received_message : received_message_list.requests()) {
      auto& received_name = received_message.tensor_name();
      ready_to_reduce.push_back(received_name);
    }
    if (received_message_list.shutdown()) {
      // Received SHUTDOWN request from the global coordinator
      state.shut_down = true;
    }

    // Local coordinators form MPI responses and forwards to local workers
    std::map<MPIDataType, std::deque<MPIResponse>> responses_by_type;

    for (auto& tensor_name : ready_to_reduce) {
      MPIResponse response =
          ConstructMPIResponse(state.local_message_table, tensor_name);
      auto& entry = state.tensor_table[response.tensor_names()[0]];
      responses_by_type[entry.tensor->dtype()].push_back(std::move(response));
    }

    MPIResponseList response_list;
    response_list.set_shutdown(state.shut_down);
    should_shut_down = state.shut_down;

    PopulateMPIResponseList(response_list, responses_by_type, state);

    // Notify all nodes which tensors we'd like to reduce at this step.
    std::string encoded_response;
    MPIResponseList::SerializeToString(response_list, encoded_response);
    int encoded_response_length = (int)encoded_response.length() + 1;

    MPI_Bcast(&encoded_response_length, 1, MPI_INT, RANK_ZERO, state.local_comm);
    MPI_Bcast((void*)encoded_response.c_str(), encoded_response_length,
              MPI_BYTE, RANK_ZERO, state.local_comm);

    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    for (auto& response : response_list.responses()) {
      PerformOperation(state.tensor_table, response);
    }

    // Check for stalled tensors.
    //if (std::chrono::steady_clock::now() - state.last_stall_check >
    //    STALL_WARNING_TIME) {
    //  CheckForStalledTensors(state);
    //  state.last_stall_check = std::chrono::steady_clock::now();
    //}
  } else {
    std::string encoded_message;
    MPIRequestList message_list;
    message_list.set_shutdown(state.shut_down);
    while (!message_queue.empty()) {
      message_list.add_requests(message_queue.front());
      message_queue.pop();
    }
    MPIRequestList::SerializeToString(message_list, encoded_message);
    int encoded_message_length = (int)encoded_message.length() + 1;
    int max_length;

    // Send messages to local node coordinator (local rank 0)
    MPI_Allreduce(&encoded_message_length, &max_length, 1, MPI_INT, MPI_MAX, state.local_comm);
    encoded_message.resize(max_length-1);
    MPI_Gather((void*)encoded_message.c_str(), max_length, MPI_BYTE, nullptr, 0, MPI_BYTE, RANK_ZERO, state.local_comm);

    // Receive instructions from local coordinator (local rank 0)
    int msg_length;
    MPI_Bcast(&msg_length, 1, MPI_INT, RANK_ZERO, state.local_comm);
    auto buffer = new char[msg_length];
    MPI_Bcast(buffer, msg_length, MPI_BYTE, RANK_ZERO, state.local_comm);
    std::string received_message(buffer, (size_t)msg_length);
    MPIResponseList response_list;
    MPIResponseList::ParseFromString(response_list, received_message);
    delete[] buffer;

    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    for (auto& response : response_list.responses()) {
      PerformOperation(state.tensor_table, response);
    }

    if (response_list.shutdown()) {
      should_shut_down = true;
    }
  }

  return !should_shut_down;
}
// Start Horovod background thread. Ensure that this is
// only done once no matter how many times this function is called.
void InitializeHorovodOnce() {
  // Ensure background thread is only started once.
  if (!horovod_global.initialize_flag.test_and_set()) {
    horovod_global.background_thread =
        std::thread(BackgroundThreadLoop, std::ref(horovod_global));
  }

  // Wait to ensure that the background thread has finished initializing MPI.
  while (!horovod_global.initialization_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

std::shared_ptr<horovod::common::OpContext> CreateMXOpContext(int device) {
  return std::make_shared<horovod::MX::MXOpContext<mxnet::NDArray>>(device, nullptr);
}
} // namespace

Status CheckInitialized() {
  if (!horovod_global.initialization_done) {
    return NOT_INITIALIZED_ERROR;
  }
  return Status::OK();
}

extern "C" {

void horovod_init() { InitializeHorovodOnce(); }

int horovod_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.rank;
}

int horovod_local_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.local_rank;
}

int horovod_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.size;
}

int horovod_local_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.local_size;
}

int horovod_mpi_threads_supported() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.mpi_threads_supported ? 1 : 0;
}
}

// MPI must be initialized and the background thread must be running before
// this function is called.
Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  MPIRequest message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(MPIRequest::ALLREDUCE);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.emplace(name, std::move(e));
    horovod_global.message_queue.push(message);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }
}

// MPI must be initialized and the background thread must be running before
// this function is called.
Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  MPIRequest message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(MPIRequest::ALLGATHER);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.emplace(name, std::move(e));
    horovod_global.message_queue.push(message);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }
}

// MPI must be initialized and the background thread must be running before
// this function is called.
Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  MPIRequest message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_root_rank(root_rank);
  message.set_device(device);
  message.set_request_type(MPIRequest::BROADCAST);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = root_rank;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.emplace(name, std::move(e));
    horovod_global.message_queue.push(message);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }
}

} // namespace common
} // namespace horovod
