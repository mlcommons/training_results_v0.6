#ifndef CC_DUAL_NET_TRT_DUAL_NET_H_
#define CC_DUAL_NET_TRT_DUAL_NET_H_

#include <memory>
#include <string>

#include "cc/dual_net/dual_net.h"

namespace minigo {

    class TrtDualNetFactory : public DualNetFactory {
    public:
        TrtDualNetFactory();
        TrtDualNetFactory(int batch_size);

        int GetBufferCount() const override;

        bool NeedCopy() override;

        std::unique_ptr<DualNet> NewDualNet(const std::string& model) override;

    private:
        int device_count_;
        int batch_size_;
    };

}  // namespace minigo

#endif  // CC_DUAL_NET_TRT_DUAL_NET_H_
