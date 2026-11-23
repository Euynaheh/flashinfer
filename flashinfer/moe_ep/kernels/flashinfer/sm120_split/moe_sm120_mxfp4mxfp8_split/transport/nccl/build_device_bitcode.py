"""Build the MegaMoE NCCL device-bridge bitcode for one NCCL version.

    python -m moe_sm120_mxfp4mxfp8_split.transport.nccl.build_device_bitcode \
        [--nccl-root DIR] [--cuda-headers DIR] [--output FILE]

CuTeDSL links external device functions only as LLVM bitcode, so the bridge is
compiled with clang in CUDA device-only mode. clang 18 parses CUDA 12.x
headers but not CUDA 13; by default this tool downloads three pinned CUDA 12.8
header wheels (sha256-checked) into ``$MEGAMOE_NCCL_CACHE/cuda128``
(default ``~/.cache/megamoe/nccl``).

The output defaults to ``prebuilt/megamoe_nccl_device_<NCCL_VERSION_CODE>.bc``
next to a ``.json`` stamp (NCCL version, source hashes, compiler) that the
runtime checks before linking. Host bridge and device bitcode must use the same
NCCL headers because the shared state embeds ``ncclDevComm``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

from .runtime import DEVICE_SOURCE, HERE, STATE_HEADER, _cache_root, _header_version, find_nccl, prebuilt_bitcode_path

# Pinned CUDA 12.8 header wheels (headers only are used).
_CUDA_HEADER_WHEELS = [
    ("nvidia-cuda-runtime-cu12", "12.8.90", "adade8dcbd0edf427b7204d480d6066d33902cab2a4707dcfc48a2d0fd44ab90"),
    ("nvidia-cuda-nvcc-cu12", "12.8.93", "2d6dc36fb7cb5ac9c0b8825bc13d193c35487a315664007287d0126531238011"),
    ("nvidia-cuda-cccl-cu12", "12.8.90", "575bc5cfb0fe4a1956c539215c0f1e6718821b95a68ef6b7abf226aa87e3a5ad"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wheel_url(name: str, version: str) -> str:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as response:
        meta = json.load(response)
    for entry in meta["urls"]:
        if entry["filename"].endswith(".whl") and "x86_64" in entry["filename"]:
            return entry["url"]
    raise RuntimeError(f"no x86_64 wheel for {name}=={version}")


def fetch_cuda_headers(cache: Path) -> Path:
    """Merged include dir from the pinned wheels (downloaded once)."""

    include = cache / "include"
    stamp = cache / "MANIFEST.json"
    if stamp.is_file() and include.is_dir():
        return include
    cache.mkdir(parents=True, exist_ok=True)
    manifest = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, version, digest in _CUDA_HEADER_WHEELS:
            url = _wheel_url(name, version)
            wheel = Path(tmp) / url.rsplit("/", 1)[1]
            urllib.request.urlretrieve(url, wheel)
            if _sha256(wheel) != digest:
                raise RuntimeError(f"{wheel.name}: sha256 mismatch")
            with zipfile.ZipFile(wheel) as archive:
                for member in archive.namelist():
                    marker = "/include/"
                    if marker not in member or member.endswith("/"):
                        continue
                    relative = member.split(marker, 1)[1]
                    target = include / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            manifest.append(dict(package=name, version=version, sha256=digest, url=url))
    stamp.write_text(json.dumps(manifest, indent=1))
    return include


def _gcc_install_dir() -> list:
    # libstdc++ 14 declares __float128 limits that nvptx clang rejects; prefer 13.
    for version in ("13", "12", "11"):
        path = Path(f"/usr/lib/gcc/x86_64-linux-gnu/{version}")
        if path.is_dir():
            return [f"--gcc-install-dir={path}"]
    return []


def _fallback_cuda_include() -> list:
    # clang's CUDA wrapper also includes curand headers, which the pinned wheels
    # do not ship; take them (only) from the local toolkit, searched last.
    home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    for candidate in (home / "targets/x86_64-linux/include", home / "include"):
        if (candidate / "curand_mtgp32_kernel.h").is_file():
            return ["-idirafter", str(candidate)]
    return []


def build(nccl_root: Path | None, cuda_headers: Path | None, output: Path | None) -> Path:
    if nccl_root is not None:
        os.environ["MEGAMOE_NCCL_ROOT"] = str(nccl_root)
        find_nccl.cache_clear()
    nccl = find_nccl()
    headers = cuda_headers or fetch_cuda_headers(_cache_root() / "cuda128")
    output = output or prebuilt_bitcode_path(nccl.version)
    output.parent.mkdir(parents=True, exist_ok=True)
    clang = shutil.which(os.environ.get("MEGAMOE_NCCL_CLANG", "clang++"))
    if clang is None:
        raise RuntimeError("clang++ is required (MEGAMOE_NCCL_CLANG)")
    command = [
        clang, "-x", "cuda", "--cuda-device-only", "--cuda-gpu-arch=sm_90", "-nocudainc", "-nocudalib",
        f"-I{headers}", "-include", "__clang_cuda_runtime_wrapper.h", *_gcc_install_dir(),
        "-U__SIZEOF_FLOAT128__", *_fallback_cuda_include(), "-Wno-unknown-cuda-version", "-O2", "-std=gnu++17",
        "-c", "-emit-llvm",
        f"-I{HERE}", f"-I{nccl.include_dir}", str(DEVICE_SOURCE), "-o", str(output),
    ]
    subprocess.run(command, check=True)
    version_line = subprocess.run([clang, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    stamp = dict(
        nccl_version=nccl.version,
        nccl_header_version=_header_version(nccl.include_dir),
        sources={DEVICE_SOURCE.name: _sha256(DEVICE_SOURCE), STATE_HEADER.name: _sha256(STATE_HEADER)},
        nccl_device_h_sha256=_sha256(nccl.include_dir / "nccl_device.h"),
        bitcode_sha256=_sha256(output),
        compiler=version_line,
        cuda_headers=(str(headers) if cuda_headers else
                      "pinned wheels: " + ", ".join(f"{n} {v}" for n, v, _ in _CUDA_HEADER_WHEELS)
                      + " (+curand from local toolkit)"),
    )
    output.with_suffix(".json").write_text(json.dumps(stamp, indent=1) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nccl-root", type=Path)
    parser.add_argument("--cuda-headers", type=Path, help="CUDA 12.x include dir (default: pinned wheels)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build(args.nccl_root, args.cuda_headers, args.output))


if __name__ == "__main__":
    main()
