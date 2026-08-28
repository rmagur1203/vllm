"""DDTree CUDA 트리 커널 로더.

vLLM 전체 재빌드 없이, 커널 하나만 별도 torch extension 으로 JIT 컴파일한다.
첫 호출에서 ~60초, 이후 캐시됨.
"""
import os
import threading

_LOCK = threading.Lock()
_EXT = None


def get_ext():
    """컴파일된 확장을 돌려준다. 실패하면 None (호출부가 Triton 으로 폴백)."""
    global _EXT
    if _EXT is not None:
        return _EXT if _EXT is not False else None
    with _LOCK:
        if _EXT is None:
            try:
                from torch.utils.cpp_extension import load
                here = os.path.dirname(os.path.abspath(__file__))
                bd = os.path.join(here, "build")
                os.makedirs(bd, exist_ok=True)
                _EXT = load(
                    name="ddtree_gdn",
                    sources=[os.path.join(here, "ddtree_gdn_tree.cu")],
                    build_directory=bd,
                    extra_cuda_cflags=[
                        "-O3", "-gencode=arch=compute_86,code=sm_86",
                        "--expt-relaxed-constexpr",
                    ],
                    extra_cflags=["-O3"],
                    verbose=False,
                )
            except Exception:
                _EXT = False
    return _EXT if _EXT is not False else None
