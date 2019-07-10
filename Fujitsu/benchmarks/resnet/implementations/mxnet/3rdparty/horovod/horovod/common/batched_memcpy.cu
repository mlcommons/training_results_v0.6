// Copyright (C) 2018 NVIDIA CORPORATION. All rights reserved.
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

#define TO_NEXT_MULT_P2(x,p)	(((x)+((p)-1)) & ~(p-1))

__host__ __device__ ulonglong2 operator<<(ulonglong2 a, int l) {
	ulonglong2 b;
	if (l > 64) {
		b = make_ulonglong2(0ull, a.x << (l-64));
	} else {
		b = make_ulonglong2(a.x << l, (a.y << l) | (a.x >> (8*sizeof(a.x)-l)));
	}
	return b; 
}

__host__ __device__ ulonglong2 operator>>(ulonglong2 a, int l) {
	ulonglong2 b;
	if (l > 64) {
		b = make_ulonglong2(a.y >> (l-64), 0ull);
	} else {
		b = make_ulonglong2((a.x >> l) | (a.y << (8*sizeof(a.y)-l)), a.y >> l);
	}
	return b;
}

__host__ __device__ ulonglong2 operator|(ulonglong2 a, ulonglong2 b) {
	return make_ulonglong2(a.x | b.x, a.y | b.y);
}

template<int BDIM_X,
	 int MAXIOB,
	 int SH_BYTE_X_BL,
	 typename LDST_T>
__device__ void memcpy_d(const size_t n,
			 const unsigned char *__restrict__ src,
			 unsigned char *__restrict__ dst,
			 unsigned char *__restrict__ __sh) {

	const int tid = threadIdx.x;

	const unsigned long long srcULL = reinterpret_cast<unsigned long long>(src);
	const unsigned long long dstULL = reinterpret_cast<unsigned long long>(dst);

	int srcOff = (MAXIOB - srcULL) & (MAXIOB-1);
	int dstOff = (MAXIOB - dstULL) & (MAXIOB-1);

	const int ELXTH = SH_BYTE_X_BL/(BDIM_X*MAXIOB);
	LDST_T *__ptrSH = reinterpret_cast<LDST_T *>(__sh);

	if (srcOff == dstOff) {

		const LDST_T *__restrict__ __ptrLDG = reinterpret_cast<const LDST_T *>(src + srcOff);
		      LDST_T *__restrict__ __ptrSTG = reinterpret_cast<      LDST_T *>(dst + dstOff);
		
		int nread    = (n-srcOff) / sizeof(*__ptrLDG);
		int remBytes = (n-srcOff) % sizeof(*__ptrLDG);

		LDST_T __loc[ELXTH];

		#pragma unroll
		for(int j = 0; j < ELXTH; j++) {
			if (j*BDIM_X+tid < nread) {
				__loc[j] = __ptrLDG[j*BDIM_X+tid];
			}
		}

		for(int i = 0; i < nread; i += BDIM_X*ELXTH) {

			#pragma unroll
			for(int j = 0; j < ELXTH; j++) {
				__ptrSH[j*BDIM_X+tid] = __loc[j];
			}
		
			#pragma unroll
			for(int j = 0; j < ELXTH; j++) {
				if (i + BDIM_X*ELXTH + j*BDIM_X + tid < nread) {
					__loc[j] = __ptrLDG[i + BDIM_X*ELXTH + j*BDIM_X + tid];
				}
			}

			#pragma unroll
			for(int j = 0; j < ELXTH; j++) {
				if (i + j*BDIM_X + tid < nread) {
					__ptrSTG[i + j*BDIM_X + tid] = __ptrSH[j*BDIM_X+tid];
				}
			}
		}
		if (tid < srcOff+remBytes) {
			const int off = (tid < srcOff) ? tid : n-remBytes+tid-srcOff;
			dst[off] = src[off];
		}
        } else {
		const LDST_T *__restrict__ __ptrLDG = reinterpret_cast<const LDST_T *>(src + srcOff);
		      LDST_T *__restrict__ __ptrSTG = reinterpret_cast<      LDST_T *>(dst + dstOff);
		
		int nread    = ((n-srcOff) / sizeof(*__ptrLDG));
		int remBytes = ((n-srcOff) % sizeof(*__ptrLDG));

		int lowShft, uppShft;
		if (srcOff > dstOff) {
			uppShft = (srcOff-dstOff)*8;
			lowShft = (8*sizeof(*__ptrLDG)) - uppShft;
			__ptrSTG++;
		} else {
			lowShft = (dstOff-srcOff)*8;
			uppShft = (8*sizeof(*__ptrLDG)) - lowShft;
		}

		for(int i = 0; i < nread-1; i += BDIM_X) {
			if (i+tid < nread-1) {
				const LDST_T low = __ptrLDG[i+tid];
				const LDST_T upp = __ptrLDG[i+tid+1];

				__ptrSTG[i+tid] = (low >> lowShft) | (upp << uppShft);
			}
		}

		remBytes += sizeof(*__ptrLDG);
		if (srcOff > dstOff) {
			dstOff += sizeof(*__ptrLDG);
			if (tid < dstOff+remBytes) {
			const int off = (tid < dstOff) ? tid : n-remBytes + tid-dstOff;
				dst[off] = src[off];
			}
		} else {
			if (tid < dstOff+remBytes) {
				const int off = (tid < dstOff) ? tid : n-remBytes + tid-dstOff;
				dst[off] = src[off];
			}
		}
	}
}

template<int BDIM_X,
	 int MAXIOB>
__global__ void memcpy_k(const size_t *sizes,
			 const unsigned char *const __restrict__ *__restrict__ in,
			 unsigned char *__restrict__ *__restrict__ out) {

	const int SH_BYTE_X_BL = 32768;
	__shared__ unsigned char __sh[SH_BYTE_X_BL];

	switch(MAXIOB) {
		case 4:
			memcpy_d<BDIM_X, MAXIOB, SH_BYTE_X_BL, unsigned int>(sizes[blockIdx.x],
									     in[blockIdx.x],
									     out[blockIdx.x],
									     __sh);
			break;
		case 8:
			memcpy_d<BDIM_X, MAXIOB, SH_BYTE_X_BL, unsigned long long>(sizes[blockIdx.x],
										   in[blockIdx.x],
										   out[blockIdx.x],
										   __sh);
			break;
		case 16:
			memcpy_d<BDIM_X, MAXIOB, SH_BYTE_X_BL, ulonglong2>(sizes[blockIdx.x],
									   in[blockIdx.x],
									   out[blockIdx.x],
									   __sh);
			break;
	}
	return;
}



#define NTHREADS 1024
void batched_d2d_memcpy(void** out_ptrs, void** in_ptrs, size_t* sizes, int num_copies, cudaStream_t stream)
{
  memcpy_k<NTHREADS, 16><<<num_copies, NTHREADS, 0, stream>>>(sizes, (unsigned char**) in_ptrs, (unsigned char**) out_ptrs);
}

