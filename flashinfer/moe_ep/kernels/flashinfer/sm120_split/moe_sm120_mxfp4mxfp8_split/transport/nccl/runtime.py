"""Locate NCCL, build/load the MegaMoE NCCL host bridge, select device bitcode.

The transport needs NCCL's device API (LSA and GIN), first available in the
version pinned by ``MIN_NCCL_VERSION``. PyTorch may already have loaded an
older NCCL; the host bridge therefore ``dlopen``s the selected library
privately and creates its own EP communicator.

Selection order for the NCCL root (directory containing ``include/`` and
``lib/libnccl.so.2``):

0. roots registered in ``PREFERRED_ROOTS`` by an embedding framework
   (FlashInfer registers its own NCCL discovery here);
1. ``MEGAMOE_NCCL_ROOT``;
2. the ``nvidia-nccl-cu13`` / ``nvidia-nccl-cu12`` wheel, if new enough.

The host bridge is compiled once per (source, NCCL headers) with ``g++`` and
cached under ``MEGAMOE_NCCL_CACHE`` (default ``~/.cache/megamoe/nccl``). The
device bridge is LLVM bitcode linked into CuTeDSL kernels. Because clang can
only parse CUDA 12.x headers, a prebuilt bitcode per NCCL version ships in
``prebuilt/``; ``build_device_bitcode`` (this package) regenerates it.
``MEGAMOE_NCCL_DEVICE_BITCODE`` overrides the path. Host and device must come
from the same NCCL version; any mismatch raises instead of falling back.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
from typing import Optional

MIN_NCCL_VERSION = 23007  # 2.30.7: device API with GIN VA signals
HERE = Path(__file__).resolve().parent
HOST_SOURCE = HERE / "megamoe_nccl_host.cc"
DEVICE_SOURCE = HERE / "megamoe_nccl_device.cu"
STATE_HEADER = HERE / "megamoe_nccl_state.h"

# FlashInfer adaptation: (origin, root) pairs tried before MEGAMOE_NCCL_ROOT,
# and the default host-bridge cache directory when MEGAMOE_NCCL_CACHE is unset.
# Both must be set before the first ``find_nccl``/``host_bridge_path`` call.
PREFERRED_ROOTS: list = []
DEFAULT_CACHE_ROOT: Optional[Path] = None


def version_string(code: int) -> str:
    return f"{code // 10000}.{code // 100 % 100}.{code % 100}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _header_version(include_dir: Path) -> int:
    for line in (include_dir / "nccl.h").read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["#define", "NCCL_VERSION_CODE"]:
            return int(parts[2])
    raise RuntimeError(f"NCCL_VERSION_CODE not found in {include_dir / 'nccl.h'}")


@dataclass(frozen=True)
class NcclInstallation:
    root: Path
    include_dir: Path
    library: Path
    version: int

    @property
    def version_str(self) -> str:
        return version_string(self.version)


def _candidate_roots():
    for origin, root in PREFERRED_ROOTS:
        yield origin, Path(root).expanduser()
    explicit = os.environ.get("MEGAMOE_NCCL_ROOT")
    if explicit:
        yield "MEGAMOE_NCCL_ROOT", Path(explicit).expanduser()
        return
    for dist_name in ("nvidia-nccl-cu13", "nvidia-nccl-cu12"):
        try:
            dist = importlib.metadata.distribution(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        yield dist_name, Path(dist.locate_file("nvidia/nccl"))


@lru_cache(maxsize=1)
def find_nccl() -> NcclInstallation:
    """Return the NCCL installation used by the transport (no fallback)."""

    rejected = []
    for origin, root in _candidate_roots():
        root = root.resolve()
        include_dir, library = root / "include", root / "lib/libnccl.so.2"
        if not library.is_file() and (root / "lib64/libnccl.so.2").is_file():
            library = root / "lib64/libnccl.so.2"
        if not (include_dir / "nccl_device.h").is_file() or not library.is_file():
            rejected.append(f"{origin}={root}: missing include/nccl_device.h or lib/libnccl.so.2")
            continue
        version = _header_version(include_dir)
        if version < MIN_NCCL_VERSION:
            rejected.append(f"{origin}={root}: NCCL {version_string(version)} < {version_string(MIN_NCCL_VERSION)}")
            continue
        return NcclInstallation(root, include_dir, library, version)
    raise RuntimeError(
        "MegaMoE NCCL transport needs NCCL >= "
        f"{version_string(MIN_NCCL_VERSION)} with device headers. Set "
        "MEGAMOE_NCCL_ROOT to a directory with include/ and lib/libnccl.so.2, "
        "or install nvidia-nccl-cu13. "
        + ("Rejected: " + "; ".join(rejected) if rejected else "No candidate found.")
    )


def _cache_root() -> Path:
    default = DEFAULT_CACHE_ROOT or Path.home() / ".cache/megamoe/nccl"
    root = Path(os.environ.get("MEGAMOE_NCCL_CACHE", default))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cuda_include_dirs():
    dirs = []
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    if (Path(home) / "include/cuda_runtime_api.h").is_file():
        dirs.append(Path(home) / "include")
    try:
        from cuda.pathfinder import find_nvidia_header_directory

        found = find_nvidia_header_directory("cudart")
        if found:
            dirs.append(Path(found))
    except Exception:
        pass
    if not dirs:
        raise RuntimeError("CUDA runtime headers not found; set CUDA_HOME")
    return dirs


def _cudart_link_args():
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    for lib in (Path(home) / "lib64", Path(home) / "targets/x86_64-linux/lib"):
        if (lib / "libcudart.so").exists():
            return [f"-L{lib}", "-lcudart", f"-Wl,-rpath,{lib}"]
    # libcudart is already loaded by PyTorch; resolve lazily at dlopen time.
    return ["-Wl,--allow-shlib-undefined"]


@lru_cache(maxsize=1)
def host_bridge_path() -> Path:
    nccl = find_nccl()
    key = hashlib.sha256(
        (_sha256(HOST_SOURCE) + _sha256(STATE_HEADER) + _sha256(nccl.include_dir / "nccl.h")
         + _sha256(nccl.include_dir / "nccl_device.h")).encode()
    ).hexdigest()[:20]
    output = _cache_root() / f"megamoe_nccl_host_{nccl.version}_{key}.so"
    if output.is_file():
        return output
    compiler = shutil.which(os.environ.get("MEGAMOE_NCCL_CXX", "g++"))
    if compiler is None:
        raise RuntimeError("g++ is required to build the MegaMoE NCCL host bridge (MEGAMOE_NCCL_CXX)")
    temporary = output.with_suffix(f".tmp.{os.getpid()}.so")
    command = [compiler, "-std=c++17", "-O2", "-shared", "-fPIC", f"-I{HERE}", f"-I{nccl.include_dir}"]
    command += [f"-I{d}" for d in _cuda_include_dirs()]
    command += [str(HOST_SOURCE), "-o", str(temporary), "-ldl"] + _cudart_link_args()
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        os.replace(temporary, output)
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"failed to build the MegaMoE NCCL host bridge:\n{' '.join(command)}\n{exc.stderr}") from exc
    return output


def prebuilt_bitcode_path(version: int) -> Path:
    return HERE / "prebuilt" / f"megamoe_nccl_device_{version}.bc"


@lru_cache(maxsize=1)
def device_bitcode() -> str:
    """Path of the device bridge bitcode matching the selected NCCL."""

    nccl = find_nccl()
    explicit = os.environ.get("MEGAMOE_NCCL_DEVICE_BITCODE")
    path = Path(explicit).expanduser().resolve() if explicit else prebuilt_bitcode_path(nccl.version)
    if not path.is_file():
        raise RuntimeError(
            f"No MegaMoE NCCL device bitcode for NCCL {nccl.version_str} ({path}). Build one with "
            "`python -m moe_sm120_mxfp4mxfp8_split.transport.nccl.build_device_bitcode` (writes "
            "prebuilt/ by default, or pass --output and set MEGAMOE_NCCL_DEVICE_BITCODE)."
        )
    stamp = path.with_suffix(".json")
    if stamp.is_file():
        import json

        meta = json.loads(stamp.read_text())
        if int(meta.get("nccl_version", -1)) != nccl.version:
            raise RuntimeError(f"device bitcode {path} was built for NCCL {meta.get('nccl_version')}, selected {nccl.version}")
        for name, digest in meta.get("sources", {}).items():
            if _sha256(HERE / name) != digest:
                raise RuntimeError(f"device bitcode {path} is stale: {name} changed; rebuild it")
    return str(path)


class HostBridge:
    """ctypes view of the host bridge for one process (one library load)."""

    def __init__(self):
        nccl = find_nccl()
        self.nccl = nccl
        lib = ctypes.CDLL(str(host_bridge_path()))
        V, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong
        lib.megamoe_nccl_error.restype = ctypes.c_char_p
        lib.megamoe_nccl_set_library.argtypes = [ctypes.c_char_p]
        lib.megamoe_nccl_version.argtypes = [ctypes.POINTER(I)]
        lib.megamoe_nccl_unique_id.argtypes = [V]
        lib.megamoe_nccl_create.argtypes = [V, I, I, I, I, I, I, ctypes.POINTER(V), ctypes.POINTER(U), ctypes.POINTER(I)]
        lib.megamoe_nccl_register.argtypes = [V, U, U]
        lib.megamoe_nccl_peer.argtypes = [V, I, I, ctypes.POINTER(U)]
        lib.megamoe_nccl_destroy.argtypes = [V]
        self.lib = lib
        self.path = host_bridge_path()
        self.check(lib.megamoe_nccl_set_library(str(nccl.library).encode()))
        version = I()
        self.check(lib.megamoe_nccl_version(ctypes.byref(version)))
        if version.value != nccl.version:
            raise RuntimeError(f"libnccl reports {version_string(version.value)} but headers are {nccl.version_str}")
        if lib.megamoe_nccl_header_version() != nccl.version:
            raise RuntimeError("host bridge was built against different NCCL headers")
        self.version = version.value

    def check(self, rc: int) -> None:
        if rc:
            raise RuntimeError(self.lib.megamoe_nccl_error().decode())


_BRIDGE: Optional[HostBridge] = None


def host_bridge() -> HostBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = HostBridge()
    return _BRIDGE


__all__ = [
    "MIN_NCCL_VERSION",
    "NcclInstallation",
    "HostBridge",
    "device_bitcode",
    "find_nccl",
    "host_bridge",
    "host_bridge_path",
    "prebuilt_bitcode_path",
]
