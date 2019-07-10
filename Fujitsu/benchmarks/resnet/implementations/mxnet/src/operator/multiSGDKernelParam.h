/*!
 *  Copyright FUJITSU LIMITED 2019
 * \file multiSGDKernelParam.h
 * \brief Optimizer parameter structure
 */

#ifndef MultiSGDKernelParam_H
#define MultiSGDKernelParam_H

#ifndef LARGE_BATCH
#define LARGE_BATCH
#endif

template<typename DType, typename MPDType>
struct MultiSGDKernelParam {
  static const int N = 55;
  int count;
  size_t max_size;
  size_t sizes[N];
  DType * weights[N];
  DType * grads[N];
  MPDType * mom[N];
  MPDType * weights32[N];
  DType * out_data[N];
  MPDType lrs[N];
  MPDType wds[N];
  MPDType clip_gradient;
  MPDType rescale_grad;
  MPDType momentum;
#ifdef LARGE_BATCH
  MPDType tmp_coeff;
  MPDType best_coeff;
  MPDType tmp_wd_coeff;
  MPDType best_wd_coeff;
  MPDType * tmp_w[N];
  MPDType eta;
  int     world_size;
  unsigned long long int     lars_flags;
  int     has_momentum;
  int     has_mixed_precision;
#endif
};


#endif
