// Host bridge for the MegaMoE NCCL transport.
//
// Loads one explicitly selected libnccl.so.2 privately (RTLD_LOCAL |
// RTLD_DEEPBIND) so it can coexist with the NCCL that PyTorch already loaded,
// creates a dedicated EP communicator with the device API (LSA, optionally
// GIN), owns the symmetric windows and publishes MegaMoeNcclState to device
// memory. Every entry point returns 0 on success and -1 with megamoe_nccl_error()
// set otherwise; nothing here falls back silently.
#define NCCL_HOSTLIB_ONLY
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "megamoe_nccl_state.h"

#define MM_STR1(x) #x
#define MM_STR(x) MM_STR1(x)

static thread_local std::string g_error;
static void* g_library = nullptr;
static std::string g_path;

template <class T>
static T nccl_symbol(const char* name) {
  if (!g_library) {
    g_library = dlopen(g_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!g_library) throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
  }
  void* p = dlsym(g_library, name);
  if (!p) throw std::runtime_error(std::string("missing NCCL symbol ") + name);
  return reinterpret_cast<T>(p);
}
static void nccl_check(ncclResult_t r, const char* what) {
  if (r != ncclSuccess) throw std::runtime_error(std::string(what) + " failed with NCCL status " + std::to_string((int)r));
}
static void cuda_check(cudaError_t r, const char* what) {
  if (r != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(r));
}
#define NCCL_CALL(name, ...) nccl_check(nccl_symbol<decltype(&name)>(MM_STR(name))(__VA_ARGS__), MM_STR(name))
#define MM_BEGIN try { g_error.clear();
#define MM_END return 0; } catch (const std::exception& e) { g_error = e.what(); return -1; }

struct Session {
  ncclComm_t comm = nullptr;
  MegaMoeNcclState state{};
  MegaMoeNcclState* device = nullptr;
  bool dev_created = false;
};

extern "C" const char* megamoe_nccl_error() { return g_error.c_str(); }
extern "C" int megamoe_nccl_state_bytes() { return (int)sizeof(MegaMoeNcclState); }
extern "C" int megamoe_nccl_header_version() { return NCCL_VERSION_CODE; }

extern "C" int megamoe_nccl_set_library(const char* path) {
  if (g_library || !path || !*path) { g_error = "library already loaded or empty path"; return -1; }
  g_path = path;
  return 0;
}

extern "C" int megamoe_nccl_version(int* out) {
  MM_BEGIN
  NCCL_CALL(ncclGetVersion, out);
  (void)nccl_symbol<decltype(&ncclDevCommCreate)>(MM_STR(ncclDevCommCreate));
  (void)nccl_symbol<decltype(&ncclCommWindowRegister)>(MM_STR(ncclCommWindowRegister));
  MM_END
}

extern "C" int megamoe_nccl_unique_id(void* uid) {
  MM_BEGIN NCCL_CALL(ncclGetUniqueId, (ncclUniqueId*)uid); MM_END
}

// torch.cuda.MemPool hooks: NCCL windows need cuMem-backed allocations.
extern "C" void* megamoe_nccl_pool_allocate(size_t bytes, int device, cudaStream_t) {
  try {
    cuda_check(cudaSetDevice(device), "pool device");
    void* ptr = nullptr;
    NCCL_CALL(ncclMemAlloc, &ptr, bytes);
    return ptr;
  } catch (const std::exception& e) { g_error = e.what(); return nullptr; }
}
extern "C" void megamoe_nccl_pool_free(void* ptr, size_t, int device, cudaStream_t) {
  try { cuda_check(cudaSetDevice(device), "pool free device"); NCCL_CALL(ncclMemFree, ptr); }
  catch (const std::exception& e) { g_error = e.what(); fprintf(stderr, "MegaMoE NCCL pool free: %s\n", e.what()); }
}

// info[0..5] = gin type, gin contexts, gin connections, net device type, rank, nranks
extern "C" int megamoe_nccl_create(void* uid, int rank, int world, int network, int gin_contexts,
                                   int gin_queue_depth, int require_gdaki, void** handle,
                                   unsigned long long* device_state, int* info) {
  MM_BEGIN
  Session* s = new Session();
  *handle = s;
  NCCL_CALL(ncclCommInitRank, &s->comm, world, *reinterpret_cast<ncclUniqueId*>(uid), rank);
  ncclDevCommRequirements req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  if (network) {
    req.ginForceEnable = true;
    req.ginSignalCount = 4 * world;
    req.ginStrongSignalsRequired = true;
    req.ginVaSignalsRequired = true;
    req.ginContextCount = gin_contexts;
    req.ginQueueDepth = gin_queue_depth;
    req.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    req.ginExclusiveContexts = false;
  }
  NCCL_CALL(ncclDevCommCreate, s->comm, &req, &s->state.comm);
  s->dev_created = true;
  ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CALL(ncclCommQueryProperties, s->comm, &props);
  info[0] = props.ginType;
  info[1] = s->state.comm.ginContextCount;
  info[2] = s->state.comm.ginConnectionCount;
  info[3] = s->state.comm.ginNetDeviceTypes[0];
  info[4] = s->state.comm.rank;
  info[5] = s->state.comm.nRanks;
  if (network && info[1] != gin_contexts)
    throw std::runtime_error("NCCL granted " + std::to_string(info[1]) + " GIN contexts, requested " + std::to_string(gin_contexts));
  if (network && require_gdaki && props.ginType != NCCL_GIN_TYPE_GDAKI)
    throw std::runtime_error("GIN is not GDAKI (type=" + std::to_string(info[0]) +
                             "); set MEGAMOE_GIN_ALLOW_PROXY=1 to accept a proxy GIN");
  cuda_check(cudaMalloc((void**)&s->device, sizeof(MegaMoeNcclState)), "state allocation");
  cuda_check(cudaMemcpy(s->device, &s->state, sizeof(MegaMoeNcclState), cudaMemcpyHostToDevice), "state publication");
  *device_state = reinterpret_cast<unsigned long long>(s->device);
  MM_END
}

// Every rank must register identical logical allocations in the same order.
extern "C" int megamoe_nccl_register(void* handle, unsigned long long address, unsigned long long bytes) {
  MM_BEGIN
  Session* s = (Session*)handle;
  int i = s->state.count;
  if (i >= MEGAMOE_NCCL_MAX_WINDOWS) throw std::runtime_error("window registry full");
  auto& w = s->state.windows[i];
  w.base = address;
  w.bytes = bytes;
  // VA signals require strict ordering.
  NCCL_CALL(ncclCommWindowRegister, s->comm, (void*)address, bytes, &w.window, NCCL_WIN_STRICT_ORDERING);
  ++s->state.count;
  cuda_check(cudaMemcpy(s->device, &s->state, sizeof(MegaMoeNcclState), cudaMemcpyHostToDevice), "window publication");
  MM_END
}

// LSA peer address of window `index` (0 when the peer is not LSA-reachable).
extern "C" int megamoe_nccl_peer(void* handle, int index, int peer, unsigned long long* address) {
  MM_BEGIN
  Session* s = (Session*)handle;
  if (index < 0 || index >= s->state.count) throw std::runtime_error("invalid window index");
  void* ptr = nullptr;
  NCCL_CALL(ncclGetPeerDevicePointer, s->state.windows[index].window, 0, peer, &ptr);
  *address = reinterpret_cast<unsigned long long>(ptr);
  MM_END
}

extern "C" int megamoe_nccl_destroy(void* handle) {
  MM_BEGIN
  Session* s = (Session*)handle;
  cuda_check(cudaDeviceSynchronize(), "final device drain");
  for (int i = s->state.count - 1; i >= 0; --i) NCCL_CALL(ncclCommWindowDeregister, s->comm, s->state.windows[i].window);
  if (s->dev_created) NCCL_CALL(ncclDevCommDestroy, s->comm, &s->state.comm);
  if (s->device) cuda_check(cudaFree(s->device), "state free");
  if (s->comm) NCCL_CALL(ncclCommDestroy, s->comm);
  delete s;
  MM_END
}
