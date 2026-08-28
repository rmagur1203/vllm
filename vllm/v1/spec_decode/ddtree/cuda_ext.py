# SPDX-License-Identifier: Apache-2.0
"""DDTree GDN 트리 커널 로더.

두 경로를 지원한다.

  1. **빌드 편입** (기본) — CMakeLists 의 소스 목록에 csrc/ddtree/ddtree_gdn_tree.cu
     가 들어가 있으면 torch.ops._C.ddtree_gdn_decode_tree_mtp 로 노출된다.
  2. **JIT 컴파일** (개발용) — VLLM_DDTREE_JIT=1 이면 커널 하나만 별도 extension
     으로 컴파일한다. vLLM 전체 재빌드 없이 커널만 고칠 때 쓴다. 첫 호출 ~60초.

둘 다 실패하면 None 을 돌려주고, 호출부는 Triton 경로로 물러선다.
"""
import os
import threading

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOCK = threading.Lock()
_EXT = None


class _OpsShim:
    """torch.ops 로 등록된 커널을 JIT 확장과 같은 모양으로 감싼다."""

    @staticmethod
    def gdn_decode_tree_mtp(*args):
        return torch.ops._C.ddtree_gdn_decode_tree_mtp(*args)


def _cuda_arch_flags() -> list[str]:
    """현재 장치의 아키텍처로 gencode 를 만든다 (sm_86 하드코딩 제거)."""
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


def _load_jit():
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(
        os.path.join(here, "..", "..", "..", "..", "csrc", "ddtree",
                     "ddtree_gdn_tree.cu"))
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    bd = os.path.join(
        os.environ.get("VLLM_DDTREE_JIT_DIR", "/tmp/vllm-ddtree-build"))
    os.makedirs(bd, exist_ok=True)
    return load(
        name="ddtree_gdn",
        sources=[src],
        build_directory=bd,
        extra_cuda_cflags=["-O3", *_cuda_arch_flags(),
                           "--expt-relaxed-constexpr"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def get_ext():
    """커널 핸들을 돌려준다. 없으면 None (호출부가 Triton 으로 폴백)."""
    global _EXT
    if _EXT is not None:
        return _EXT if _EXT is not False else None
    with _LOCK:
        if _EXT is None:
            _EXT = False
            if not os.environ.get("VLLM_DDTREE_JIT"):
                if hasattr(torch.ops._C, "ddtree_gdn_decode_tree_mtp"):
                    _EXT = _OpsShim
                else:
                    logger.debug(
                        "DDTree: 빌드에 편입된 GDN 트리 커널이 없다. "
                        "VLLM_DDTREE_JIT=1 로 JIT 컴파일하거나 Triton 경로를 쓴다.")
            else:
                try:
                    _EXT = _load_jit()
                except Exception as e:      # 컴파일 실패는 치명적이지 않다
                    logger.warning("DDTree: GDN 트리 커널 JIT 실패 (%s). "
                                   "Triton 경로로 물러선다.", e)
    return _EXT if _EXT is not False else None
