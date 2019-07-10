#include "cc/dual_net/trt_dual_net.h"

#include <algorithm>
#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/logging.h"
#include "cc/thread_safe_queue.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#if MINIGO_ENABLE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/stream_executor/platform.h"
#endif

using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace minigo {
namespace {

class TrtDualNet : public DualNet {
public:
    TrtDualNet(std::string graph_path, int batch_size);

    ~TrtDualNet() override;

    void RunMany(std::vector<const BoardFeatures*> features,
                 std::vector<Output*> outputs, std::string* model) override;
    float* GetInput() override {
        return inputs_[0].second.flat<float>().data();
    }
    Output* GetOutput() override {
        return output_;
    }

private:
    std::string graph_path_;
    std::unique_ptr<Session> session_;
    std::vector<std::pair<std::string, Tensor>> inputs_;
    std::vector<std::string> output_names_;
    std::vector<Tensor> outputs_;
    size_t batch_capacity_;
    Output* output_;
};

TrtDualNet::TrtDualNet(std::string graph_path, int batch_size)
    : DualNet(std::string(file::Stem(graph_path))),
      graph_path_(graph_path) {
    GraphDef graph_def;

    auto* env = Env::Default();
    TF_CHECK_OK(ReadBinaryProto(env, graph_path, &graph_def));

    SessionOptions options;
    options.config.mutable_gpu_options()->set_allow_growth(true);
    session_.reset(NewSession(options));
    TF_CHECK_OK(session_->Create(graph_def));

    output_names_.emplace_back("policy_output");
    output_names_.emplace_back("value_output");

    // fixed batchsize
    batch_capacity_ = batch_size;
    inputs_.emplace_back(
        "pos_tensor",
        Tensor(DT_FLOAT, TensorShape({static_cast<int>(batch_capacity_), kN, kN, kNumStoneFeatures})));
    output_ = (Output*) malloc(batch_capacity_ * sizeof(Output));
}

TrtDualNet::~TrtDualNet() {
    if (session_ != nullptr) {
        TF_CHECK_OK(session_->Close());
    }
}

void TrtDualNet::RunMany(std::vector<const BoardFeatures*> features,
                        std::vector<Output*> outputs, std::string* model) {

    //auto* feature_data = inputs_[0].second.flat<float>().data();
    // Copy the features into the input tensor.
    //for (const auto* feature : features) {
    //    feature_data = std::copy(feature->begin(), feature->end(), feature_data);
    //}

    // Input should already be ready here
    // Run the model.
    TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));

    // Copy the policy and value out of the output tensors.
    // TODO: a way to pre-allocate this?
    const auto& policy_tensor = outputs_[0].flat<float>();
    const auto& value_tensor = outputs_[1].flat<float>();

    for (size_t i = 0; i < batch_capacity_; ++i) {
        memcpy(output_[i].policy.data(), policy_tensor.data() + i * kNumMoves,
               sizeof(output_[i].policy));
        output_[i].value = value_tensor.data()[i];
    }

    if (model != nullptr) {
        *model = graph_path_;
    }
}
}  // namespace

TrtDualNetFactory::TrtDualNetFactory() : device_count_(0) {
#if MINIGO_ENABLE_GPU
  if (tensorflow::ValidateGPUMachineManager().ok()) {
    device_count_ = tensorflow::GPUMachineManager()->VisibleDeviceCount();
  }
#endif
}

TrtDualNetFactory::TrtDualNetFactory(int batch_size) : device_count_(0) {
#if MINIGO_ENABLE_GPU
  if (tensorflow::ValidateGPUMachineManager().ok()) {
    device_count_ = tensorflow::GPUMachineManager()->VisibleDeviceCount();
  }
#endif
  batch_size_ = batch_size;
}

int TrtDualNetFactory::GetBufferCount() const {
    // assume always 1 buffer now
    // maybe 2 later
    return 1; //std::max(device_count_, 1);
}

bool TrtDualNetFactory::NeedCopy() {
    return false;
}

std::unique_ptr<DualNet> TrtDualNetFactory::NewDualNet(
    const std::string& model) {
    // only handle 1 device(due to TRT anyway)
    // code batchsize to be 128 now
    MG_DCHECK(device_count_ == 1);
    return absl::make_unique<TrtDualNet>(model, batch_size_);
}

}  // namespace minigo
