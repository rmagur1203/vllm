# SPDX-License-Identifier: Apache-2.0
"""DDTree — 확산 드래프트 트리 기반 추측 디코딩 (arXiv:2604.12989).

블록 디퓨전 드래프터의 위치별 분포로 best-first 트리를 만들고, 조상만 보이는
어텐션 마스크로 한 번의 forward 에 여러 후보를 동시에 검증한다.

참조 구현: https://github.com/liranringel/ddtree (MIT)
"""

from vllm.v1.spec_decode.ddtree.tree import (Tree, build_tree, build_tree_from_logits,
                                       flat_tree_mask, follow_tree,
                                       topk_from_logits)
from vllm.v1.spec_decode.ddtree.compact import (attention_caches, compact_kv_torch,
                                          compact_kv_triton)
from vllm.v1.spec_decode.ddtree.gdn import (COMPACT_STATS, compact_gdn, register_state,
                                      tree_cols_tensor, tree_conv_columns)
from vllm.v1.spec_decode.ddtree.runtime import DDTreeRuntime

__all__ = ["COMPACT_STATS", "DDTreeRuntime", "Tree", "attention_caches", "build_tree", "build_tree_from_logits", "compact_gdn", "compact_kv_torch", "compact_kv_triton", "flat_tree_mask", "follow_tree", "topk_from_logits", "register_state", "tree_cols_tensor", "tree_conv_columns"]
