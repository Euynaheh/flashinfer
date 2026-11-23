// Shared host/device state for the MegaMoE NCCL (LSA + GIN) transport.
// The layout embeds ncclDevComm by value, so the host bridge and the device
// bitcode must be built from the same NCCL headers (checked at load time).
#pragma once
#include <nccl_device.h>

#define MEGAMOE_NCCL_MAX_WINDOWS 64
#define MEGAMOE_NCCL_SHADOW_SLOTS 4096
#define MEGAMOE_NCCL_PENDING_SLOTS 2560

struct MegaMoeNcclWindow {
  unsigned long long base, bytes;
  ncclWindow_t window;
};
struct MegaMoeNcclPending {
  int peer, valid;
  unsigned long long sequence;
};
struct MegaMoeNcclState {
  ncclDevComm comm;
  int count;
  MegaMoeNcclWindow windows[MEGAMOE_NCCL_MAX_WINDOWS];
  // Commit-record shadows: one entry per (source record address). Strong VA
  // signal-add publishes the delta between the new and the last record value.
  unsigned long long shadow_keys[MEGAMOE_NCCL_SHADOW_SLOTS];
  unsigned long long shadow_values[MEGAMOE_NCCL_SHADOW_SLOTS];
  unsigned long long shadow_dst[MEGAMOE_NCCL_SHADOW_SLOTS];
  int shadow_peer[MEGAMOE_NCCL_SHADOW_SLOTS];
  int shadow_context[MEGAMOE_NCCL_SHADOW_SLOTS];
  MegaMoeNcclPending pending[MEGAMOE_NCCL_PENDING_SLOTS];
};
