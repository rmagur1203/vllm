// DDTree — GDN 융합 디코드 커널의 트리판.
//
// 원본: vllm/csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu (a9a17e709)
//
// 원본의 두 가지 8토큰 제약을 없앤다:
//   1) 공유메모리가 kMaxMtpTokens 에 비례 (토큰당 1,544B, 정적 48KB 한도)
//   2) 프리로드가 '워프 하나당 토큰 하나' (8워프)
// → 토큰을 kTile(=8) 단위로 타일링하고, 상태는 공유메모리에 두지 않는다.
//
// 트리 지원: 각 토큰이 '직전 토큰' 이 아니라 '부모 노드' 의 상태에서 출발한다.
//   부모 슬롯 = state_indices[parent[t]] (parent < 0 이면 source_slot)
//   힙 순서라 parent(t) < t 가 보장되고, 워프 w 는 자기 행만 읽고 쓰므로
//   워프 간 동기화가 필요 없다 (같은 스레드가 쓴 주소를 그대로 읽는다).
//
// 곱셈-누적 순서는 원본과 동일 → 사슬 부모를 주면 원본과 같은 결과여야 한다.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace {

template <typename StateT>
__device__ __forceinline__ float4 load_state4(const StateT* s);
template <>
__device__ __forceinline__ float4 load_state4<float>(const float* s) {
  return *reinterpret_cast<const float4*>(s);
}
template <>
__device__ __forceinline__ float4 load_state4<__nv_bfloat16>(const __nv_bfloat16* s) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(s);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(s + 2);
  return make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y),
                     __bfloat162float(hi.x), __bfloat162float(hi.y));
}

template <typename StateT>
__device__ __forceinline__ void store_state4(StateT* s, float4 v);
template <>
__device__ __forceinline__ void store_state4<float>(float* s, float4 v) {
  *reinterpret_cast<float4*>(s) = v;
}
template <>
__device__ __forceinline__ void store_state4<__nv_bfloat16>(__nv_bfloat16* s, float4 v) {
  *reinterpret_cast<__nv_bfloat162*>(s) = __floats2bfloat162_rn(v.x, v.y);
  *reinterpret_cast<__nv_bfloat162*>(s + 2) = __floats2bfloat162_rn(v.z, v.w);
}

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kTile = kWarps;           // 타일당 토큰 수 (프리로드가 워프당 1토큰)
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;

struct Strides {
  int64_t mixed_row, a_row, b_row, gate_row, state_slot;
};

__device__ __forceinline__ float sigmoid_fast(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_fast(float x) { return x * sigmoid_fast(x); }
__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float load_dt_bias(const void* p, int head, int type) {
  if (type == kDtBiasBFloat16) return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[head]);
  if (type == kDtBiasFloat16) return __half2float(static_cast<const __half*>(p)[head]);
  return static_cast<const float*>(p)[head];
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
struct Sum2 { float x, y; };
__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, o);
    y += __shfl_xor_sync(0xffffffffu, y, o);
  }
  return {x, y};
}

template <typename StateT, int ValueHeadsPerKeyHead>
__global__ __launch_bounds__(kThreads, 2) void gdn_tree_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens,
    const int* __restrict__ tree_parents,
    StateT* __restrict__ state, const __nv_bfloat16* __restrict__ output_gate,
    const void* __restrict__ norm_weight, __nv_bfloat16* __restrict__ out,
    int H, int HV, int siw, int dt_bias_type, bool norm_weight_is_bf16,
    float scale, float norm_eps, Strides strides) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bos = cu_seqlens[request], eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) return;

  const int accepted = num_accepted_tokens[request];
  const int source_slot = (accepted > 0 && accepted <= siw)
                              ? state_indices[request * siw + accepted - 1] : 0;
  if (source_slot <= 0) {
    for (int l = tid; l < num_tokens * kDimV; l += kThreads) {
      const int token = bos + l / kDimV, value = l % kDimV;
      out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] =
          __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ float shared_q[kTile][kDimK];
  __shared__ float shared_k[kTile][kDimK];
  __shared__ __nv_bfloat16 shared_v[kTile][kDimV];
  __shared__ __nv_bfloat16 shared_out[kTile][kDimV];
  __shared__ float shared_decay[kTile];
  __shared__ float shared_beta[kTile];

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) rows[r] = warp + r * kWarps;

  for (int tile = 0; tile < num_tokens; tile += kTile) {
    const int tlen = min(kTile, num_tokens - tile);
    __syncthreads();
    if (warp < tlen) {
      const int tl = warp, token = bos + tile + tl;
      const int64_t mb = static_cast<int64_t>(token) * strides.mixed_row;
      float qv[4], kv[4], qs = 0.0f, ks = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        qv[i] = __bfloat162float(mixed_qkv[mb + key_head * kDimK + d]);
        kv[i] = __bfloat162float(mixed_qkv[mb + H * kDimK + key_head * kDimK + d]);
        shared_v[tl][d] = mixed_qkv[mb + 2 * H * kDimK + value_head * kDimV + d];
        qs += qv[i] * qv[i];
        ks += kv[i] * kv[i];
      }
      const Sum2 s = warp_reduce_sum_pair(qs, ks);
      const float q_scale = __shfl_sync(0xffffffffu,
          lane == 0 ? rsqrtf(s.x + 1.0e-6f) * scale : 0.0f, 0);
      const float k_scale = __shfl_sync(0xffffffffu,
          lane == 0 ? rsqrtf(s.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        shared_q[tl][d] = qv[i] * q_scale;
        shared_k[tl][d] = kv[i] * k_scale;
      }
      if (lane == 0) {
        const float av = __bfloat162float(a[static_cast<int64_t>(token) * strides.a_row + value_head]);
        const float bv = __bfloat162float(b[static_cast<int64_t>(token) * strides.b_row + value_head]);
        const float g = -__expf(a_log[value_head]) *
                        softplus_fast(av + load_dt_bias(dt_bias, value_head, dt_bias_type));
        shared_decay[tl] = __expf(g);
        shared_beta[tl] = sigmoid_fast(bv);
      }
    }
    __syncthreads();

#pragma unroll
    for (int chunk = 0; chunk < kNumChunks; ++chunk) {
      for (int tl = 0; tl < tlen; ++tl) {
        const int t = tile + tl;
        const int par = tree_parents[request * siw + t];
        const int par_slot = (par < 0) ? source_slot : state_indices[request * siw + par];
        const StateT* src = state + static_cast<int64_t>(par_slot) * strides.state_slot +
                            value_head * kDimV * kDimK;
        float h[kRowsPerWarp][4];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
          const float4 sv = load_state4<StateT>(src + (chunk * kChunkV + rows[r]) * kDimK + k_base);
          h[r][0] = sv.x; h[r][1] = sv.y; h[r][2] = sv.z; h[r][3] = sv.w;
        }
        const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[tl][k_base]);
        const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[tl][k_base]);
        const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
        const float kk[4] = {k4.x, k4.y, k4.z, k4.w};

        float dot_hk[kRowsPerWarp] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            h[r][i] *= shared_decay[tl];
            dot_hk[r] += h[r][i] * kk[i];
          }
        const Sum2 hk01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
        const Sum2 hk23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
        const float rhk[kRowsPerWarp] = {hk01.x, hk01.y, hk23.x, hk23.y};

        float dot_hq[kRowsPerWarp] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
          const int value = chunk * kChunkV + rows[r];
          const float delta = (__bfloat162float(shared_v[tl][value]) - rhk[r]) * shared_beta[tl];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            h[r][i] += kk[i] * delta;
            dot_hq[r] += h[r][i] * qq[i];
          }
        }
        const Sum2 hq01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
        const Sum2 hq23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
        if (lane == 0) {
          shared_out[tl][chunk * kChunkV + rows[0]] = __float2bfloat16(hq01.x);
          shared_out[tl][chunk * kChunkV + rows[1]] = __float2bfloat16(hq01.y);
          shared_out[tl][chunk * kChunkV + rows[2]] = __float2bfloat16(hq23.x);
          shared_out[tl][chunk * kChunkV + rows[3]] = __float2bfloat16(hq23.y);
        }
        const int dst = state_indices[request * siw + t];
        if (dst > 0) {
          StateT* d = state + static_cast<int64_t>(dst) * strides.state_slot +
                      value_head * kDimV * kDimK;
#pragma unroll
          for (int r = 0; r < kRowsPerWarp; ++r) {
            const int value = chunk * kChunkV + rows[r];
            store_state4<StateT>(d + value * kDimK + k_base,
                                 make_float4(h[r][0], h[r][1], h[r][2], h[r][3]));
          }
        }
      }
    }
    __syncthreads();

    if (warp < tlen) {
      const int tl = warp, token = bos + tile + tl;
      float ov[4], ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int value = lane + i * 32;
        ov[i] = __bfloat162float(shared_out[tl][value]);
        ss += ov[i] * ov[i];
      }
      ss = warp_reduce_sum(ss);
      const float rstd = rsqrtf(ss / static_cast<float>(kDimV) + norm_eps);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int value = lane + i * 32;
        const float gate = silu_fast(__bfloat162float(
            output_gate[static_cast<int64_t>(token) * strides.gate_row +
                        value_head * kDimV + value]));
        const float w = norm_weight_is_bf16
            ? __bfloat162float(static_cast<const __nv_bfloat16*>(norm_weight)[value])
            : static_cast<const float*>(norm_weight)[value];
        out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] =
            __float2bfloat16(ov[i] * rstd * w * gate);
      }
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// 런처 + 바인딩
// ---------------------------------------------------------------------------
#define DDT_DISPATCH_VHPK(VHPK, ...)                       \
  switch (VHPK) {                                          \
    case 1: { constexpr int V = 1; __VA_ARGS__; break; }    \
    case 2: { constexpr int V = 2; __VA_ARGS__; break; }    \
    case 3: { constexpr int V = 3; __VA_ARGS__; break; }    \
    case 4: { constexpr int V = 4; __VA_ARGS__; break; }    \
    case 8: { constexpr int V = 8; __VA_ARGS__; break; }    \
    default: TORCH_CHECK(false, "HV/H must be in {1,2,3,4,8}, got ", VHPK); \
  }

void gdn_decode_tree_mtp(
    at::Tensor mixed_qkv, at::Tensor a, at::Tensor b, at::Tensor A_log,
    at::Tensor dt_bias, at::Tensor state_indices, at::Tensor cu_seqlens,
    at::Tensor num_accepted_tokens, at::Tensor tree_parents, at::Tensor state,
    at::Tensor output_gate, at::Tensor norm_weight, at::Tensor out,
    double scale, double norm_eps) {
  TORCH_CHECK(mixed_qkv.dim() == 2, "mixed_qkv must be [L, 2*H*128 + HV*128]");
  TORCH_CHECK(state.dim() == 4 && state.size(2) == kDimV && state.size(3) == kDimK,
              "state must be [slots, HV, 128, 128]");
  TORCH_CHECK(tree_parents.dtype() == at::kInt && tree_parents.is_cuda(),
              "tree_parents must be a CUDA int32 tensor");
  TORCH_CHECK(tree_parents.sizes() == state_indices.sizes(),
              "tree_parents must have the same shape as state_indices");

  const int HV = static_cast<int>(state.size(1));
  const int64_t key_width = mixed_qkv.size(1) - static_cast<int64_t>(HV) * kDimV;
  TORCH_CHECK(key_width > 0 && key_width % (2 * kDimK) == 0,
              "mixed_qkv width inconsistent with state");
  const int H = static_cast<int>(key_width / (2 * kDimK));
  const int vhpk = HV / H;
  const int N = static_cast<int>(state_indices.size(0));
  const int siw = static_cast<int>(state_indices.size(1));
  // 🔴 원본은 siw <= 8 로 막혀 있다 (공유메모리 + 워프당 1토큰).
  //    여기서는 토큰을 타일링하고 상태를 공유메모리에 두지 않으므로 상한이 없다.

  const int dt_type = dt_bias.scalar_type() == at::kFloat  ? kDtBiasFloat32
                      : dt_bias.scalar_type() == at::kBFloat16 ? kDtBiasBFloat16
                                                               : kDtBiasFloat16;
  const Strides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0),
                        output_gate.stride(0), state.stride(0)};
  const dim3 grid(N, HV);
  auto stream = c10::cuda::getCurrentCUDAStream();

  DDT_DISPATCH_VHPK(vhpk, {
    if (state.scalar_type() == at::kFloat) {
      gdn_tree_kernel<float, V><<<grid, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),
          static_cast<const __nv_bfloat16*>(a.data_ptr()),
          static_cast<const __nv_bfloat16*>(b.data_ptr()),
          A_log.data_ptr<float>(), dt_bias.data_ptr(),
          state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
          num_accepted_tokens.data_ptr<int>(), tree_parents.data_ptr<int>(),
          static_cast<float*>(state.data_ptr()),
          static_cast<const __nv_bfloat16*>(output_gate.data_ptr()),
          norm_weight.data_ptr(),
          static_cast<__nv_bfloat16*>(out.data_ptr()), H, HV, siw, dt_type,
          norm_weight.scalar_type() == at::kBFloat16,
          static_cast<float>(scale), static_cast<float>(norm_eps), strides);
    } else {
      gdn_tree_kernel<__nv_bfloat16, V><<<grid, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),
          static_cast<const __nv_bfloat16*>(a.data_ptr()),
          static_cast<const __nv_bfloat16*>(b.data_ptr()),
          A_log.data_ptr<float>(), dt_bias.data_ptr(),
          state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
          num_accepted_tokens.data_ptr<int>(), tree_parents.data_ptr<int>(),
          static_cast<__nv_bfloat16*>(state.data_ptr()),
          static_cast<const __nv_bfloat16*>(output_gate.data_ptr()),
          norm_weight.data_ptr(),
          static_cast<__nv_bfloat16*>(out.data_ptr()), H, HV, siw, dt_type,
          norm_weight.scalar_type() == at::kBFloat16,
          static_cast<float>(scale), static_cast<float>(norm_eps), strides);
    }
  });
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_decode_tree_mtp", &gdn_decode_tree_mtp,
        "GDN decode MTP post-conv with tree parents (no 8-token cap)");
}
