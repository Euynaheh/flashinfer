// Device bridge for the MegaMoE NCCL transport, linked into CuTeDSL kernels as
// LLVM bitcode (cute.ffi). Built from the same NCCL headers as the host bridge;
// see tools/build_nccl_device_bitcode.py.
#include <cuda_runtime.h>
#include "megamoe_nccl_state.h"

using U = unsigned long long;
using Net = ncclGin;

__device__ __forceinline__ int megamoe_find_window(MegaMoeNcclState* s, U ptr, U bytes) {
  for (int i = 0; i < s->count; ++i)
    if (ptr >= s->windows[i].base && ptr - s->windows[i].base + bytes <= s->windows[i].bytes) return i;
  asm volatile("trap;");
  return -1;
}

__device__ __forceinline__ unsigned megamoe_shadow_slot(MegaMoeNcclState* s, U key) {
  unsigned index = (key >> 3 ^ key >> 17) & (MEGAMOE_NCCL_SHADOW_SLOTS - 1), tries = 0;
  while (true) {
    U prev = atomicCAS(s->shadow_keys + index, 0ull, key);
    if (prev == 0 || prev == key) return index;
    index = (index + 1) & (MEGAMOE_NCCL_SHADOW_SLOTS - 1);
    if (++tries == MEGAMOE_NCCL_SHADOW_SLOTS) { asm volatile("trap;"); return 0; }
  }
}

// GIN put. signal_record=1 publishes an 8-byte commit record as a strong VA
// signal-add of (new - previous) value; each record source has one writer.
extern "C" __device__ __noinline__ void megamoe_gin_put(U state_ptr, int context, int slot, int peer, U dst, U src,
                                                        U bytes, int blocking, int signal_record) {
  auto* s = (MegaMoeNcclState*)state_ptr;
  Net gin(s->comm, context);
  auto team = ncclTeamWorld(s->comm);
  int d = megamoe_find_window(s, dst, bytes), a = megamoe_find_window(s, src, bytes);
  if (signal_record) {
    if (bytes != 8) { asm volatile("trap;"); return; }
    unsigned index = megamoe_shadow_slot(s, src);
    if (s->shadow_dst[index] == 0) { s->shadow_dst[index] = dst; s->shadow_peer[index] = peer; s->shadow_context[index] = context; }
    if (s->shadow_dst[index] != dst || s->shadow_peer[index] != peer || s->shadow_context[index] != context) {
      asm volatile("trap;");
      return;
    }
    U value = *(const U*)src, previous = s->shadow_values[index];
    gin.signal(team, peer, ncclGin_StrongVASignalAdd{s->windows[d].window, dst - s->windows[d].base, value - previous});
    s->shadow_values[index] = value;
  } else {
    gin.put(team, peer, s->windows[d].window, dst - s->windows[d].base, s->windows[a].window, src - s->windows[a].base, bytes);
  }
  s->pending[slot].peer = peer;
  s->pending[slot].valid = 1;
  if (blocking) gin.flush(ncclCoopThread());
}

// Local source retirement only (not a remote quiet): remote consumption and
// buffer reuse are proven by the protocol's receive commits / ACKs / credits.
extern "C" __device__ __noinline__ void megamoe_gin_flush(U state_ptr, int context, int slot) {
  auto* s = (MegaMoeNcclState*)state_ptr;
  Net gin(s->comm, context);
  gin.flush(ncclCoopThread());
  s->pending[slot].valid = 0;
}

// Payload put fused with a strong VA commit signal (record at signal_src).
extern "C" __device__ __noinline__ void megamoe_gin_put_commit(U state_ptr, int context, int slot, int peer, U dst, U src,
                                                               U bytes, U signal_dst, U signal_src) {
  auto* s = (MegaMoeNcclState*)state_ptr;
  Net gin(s->comm, context);
  auto team = ncclTeamWorld(s->comm);
  int d = megamoe_find_window(s, dst, bytes), a = megamoe_find_window(s, src, bytes), sig = megamoe_find_window(s, signal_dst, 8);
  unsigned index = megamoe_shadow_slot(s, signal_src);
  U value = *(const U*)signal_src, previous = s->shadow_values[index];
  gin.put(team, peer, s->windows[d].window, dst - s->windows[d].base, s->windows[a].window, src - s->windows[a].base, bytes,
          ncclGin_StrongVASignalAdd{s->windows[sig].window, signal_dst - s->windows[sig].base, value - previous});
  s->shadow_values[index] = value;
  s->pending[slot].peer = peer;
  s->pending[slot].valid = 1;
}

extern "C" __device__ __noinline__ void megamoe_gin_wait_signal(U state, U addr, U expected, int bits) {
  auto* s = (MegaMoeNcclState*)state;
  int d = megamoe_find_window(s, addr, 8);
  Net gin(s->comm, 0);
  gin.waitSignal(ncclCoopThread(), s->windows[d].window, addr - s->windows[d].base, expected, bits, cuda::memory_order_acquire);
}

extern "C" __device__ __noinline__ U megamoe_gin_read_signal(U state, U addr) {
  auto* s = (MegaMoeNcclState*)state;
  int d = megamoe_find_window(s, addr, 8);
  Net gin(s->comm, 0);
  return gin.readSignal(s->windows[d].window, addr - s->windows[d].base, 64, cuda::memory_order_acquire);
}

// Whole-CTA LSA copy into a peer window; system fence before returning.
extern "C" __device__ __noinline__ void megamoe_lsa_put(U state, int peer, U dst, U src, U bytes) {
  auto* s = (MegaMoeNcclState*)state;
  int d = megamoe_find_window(s, dst, bytes);
  char* target = (char*)ncclGetPeerPointer(s->windows[d].window, dst - s->windows[d].base, peer);
  if (!target) { asm volatile("trap;"); return; }
  __syncthreads();
  if (((dst | src | bytes) & 15) == 0) {
    for (U i = threadIdx.x; i < bytes / 16; i += blockDim.x) ((uint4*)target)[i] = ((const uint4*)src)[i];
  } else if (((dst | src | bytes) & 7) == 0) {
    for (U i = threadIdx.x; i < bytes / 8; i += blockDim.x) ((volatile U*)target)[i] = ((const U*)src)[i];
  } else {
    for (U i = threadIdx.x; i < bytes; i += blockDim.x) target[i] = ((const char*)src)[i];
  }
  __threadfence_system();
  __syncthreads();
}

extern "C" __device__ __noinline__ void megamoe_system_fence() { __threadfence_system(); }
