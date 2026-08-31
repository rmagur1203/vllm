# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import os
from typing import Literal

import torch
from einops import rearrange
from torch import nn

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops import (
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

# Optional ROCm AITER Triton kernels for the GDN decode path.
# Availability is checked centrally via rocm_aiter_ops; the actual function
# references are imported here so that they can be called without per-call
# import overhead.
GDN_AITER_TRITON_AVAILABLE = (
    rocm_aiter_ops.are_gdn_triton_kernels_available()
    or rocm_aiter_ops.is_rdna_gdn_triton_kernels_available()
)

if GDN_AITER_TRITON_AVAILABLE:
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token as gdn_aiter_fused_reshape_causal_conv1d_update_single_token,  # noqa: E501
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule,  # noqa: E501
    )

logger = init_logger(__name__)

# --------------------------------------------------------------------------
# DDTree: 하이브리드 타깃의 재귀 계층을 트리로 만들기 위한 훅.
#
# 재귀 계층은 상태가 주어진 순서대로 전진하므로, 트리 노드를 선형 시퀀스로 넣으면
# 형제끼리 상태가 섞인다. 어텐션 마스크로는 막을 수 없다.
#   - conv : 커널을 '수정' 해 윈도를 조상 열에서 읽게 한다 (tree_cols 인자)
#   - SSM  : 커널이 부모 슬롯의 상태를 재적재 (tree_parent_indices 인자)
# 둘 다 대체가 아니라 수정이라 사슬에서 비트 단위로 동일하다.
# --------------------------------------------------------------------------
_DDTREE_GDN_PROVIDER = None


_DDTREE_KERNEL_READY = None


def _ddtree_kernel_ready() -> bool:
    """우리 GDN 트리 커널을 실제로 쓸 수 있는지 (한 번만 확인해 캐시)."""
    global _DDTREE_KERNEL_READY
    if _DDTREE_KERNEL_READY is None:
        try:
            from vllm.v1.spec_decode.ddtree.cuda_ext import get_ext
            _DDTREE_KERNEL_READY = get_ext() is not None
        except Exception:
            _DDTREE_KERNEL_READY = False
    return _DDTREE_KERNEL_READY


def set_ddtree_gdn_provider(fn) -> None:
    """fn(n_spec_reqs, T) -> {"parents","keys","tree_cols"} | None"""
    global _DDTREE_GDN_PROVIDER
    _DDTREE_GDN_PROVIDER = fn


def _ddtree_active() -> bool:
    """DDTree 공급자가 붙어 있는가.

    🔴 이걸 먼저 봐야 한다. 아래 트리 정보 준비는 cu_seqlens / state_indices 를
       CPU 로 내리는데(계층당 2회 D2H), DDTree 를 안 쓰는 사용자에게 그 비용을
       지우면 안 된다. GDN 계층이 24개라 스텝당 48회다.
    """
    return _DDTREE_GDN_PROVIDER is not None


def _ddtree_gdn(n_spec_reqs: int, T: int, width: int = 4):
    if _DDTREE_GDN_PROVIDER is None:
        return None
    return _DDTREE_GDN_PROVIDER(n_spec_reqs, T, width)

MAX_FUSED_GDN_MTP_TOKENS = 8
FUSED_GDN_STATE_DTYPES = (torch.float32, torch.bfloat16)


def _resolve_gdn_prefill_backend(
    vllm_config: VllmConfig,
) -> tuple[str, Literal["triton", "flashinfer", "cutedsl"]]:
    """Resolve GDN prefill backend.

    FlashInfer's GDN prefill kernel is chosen when:
    * ``requested in ["flashinfer", "auto"]``;
    * ``platform == cuda``;
    * one of the following:
      - Hopper (SM90) — no further constraints;
      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``.

    In-tree CuteDSL GDN prefill kernel is chosen when:
    * "cutedsl" is requested; (opt-in only)
    * Blackwell (SM10.x) with ``head_k_dim == 128``;
    """
    additional_config = vllm_config.additional_config
    backend_cfg = (
        additional_config.get("gdn_prefill_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    backend = str(backend_cfg).strip().lower()

    if not current_platform.is_cuda():
        return backend, "triton"

    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )

    supports_flashinfer = False
    supports_cutedsl = False

    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True

    if backend in ["flashinfer", "auto"] and supports_flashinfer:
        return backend, "flashinfer"
    if backend == "cutedsl" and supports_cutedsl:
        return backend, "cutedsl"
    return backend, "triton"


def _log_gdn_backend_decision(
    vllm_config: VllmConfig,
    requested_backend: str,
    active_backend: str,
) -> None:
    """Log the GDN prefill backend choice in the attention-selector style."""
    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )
    chosen = {
        "flashinfer": "FlashInfer",
        "cutedsl": "CuteDSL",
        "triton": "Triton/FLA",
    }[active_backend]
    logger.info_once(
        "Using %s GDN prefill kernel (requested=%s, head_k_dim=%s).",
        chosen,
        requested_backend,
        head_k_dim,
    )
    if active_backend == "flashinfer" and current_platform.is_device_capability(90):
        logger.warning_once(
            "FlashInfer GDN prefill is JIT-compiled; first run may take a "
            "while. Set --gdn-prefill-backend triton to skip JIT.",
        )


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # FlashInfer returns (output, state) when output_final_state=True,
    # or just output when output_final_state=False.
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        backend, active_backend = _resolve_gdn_prefill_backend(vllm_config)
        self.gdn_prefill_backend = active_backend

        if backend in ("flashinfer", "cutedsl") and active_backend != backend:
            logger.warning_once(
                "GDN prefill backend '%s' is selected but cannot use this "
                "kernel on the current platform. Falling back to Triton/FLA.",
                backend,
            )
        _log_gdn_backend_decision(vllm_config, backend, active_backend)

        if active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
        elif active_backend == "cutedsl":
            self._forward_method = self.forward_cutedsl
        else:
            self._forward_method = self.forward_native

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )

    def forward_cutedsl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
            chunk_gated_delta_rule_cutedsl,
        )

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        assert cu_seqlens is not None
        assert chunk_indices is not None
        assert chunk_offsets is not None

        o, final_state = chunk_gated_delta_rule_cutedsl(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        if not output_final_state:
            final_state = None
        return o, final_state


@PluggableLayer.register("qwen_gated_delta_net_attention")
class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        gqa_interleaved_layout=False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.gqa_interleaved_layout = gqa_interleaved_layout
        if current_platform.is_xpu():
            self._forward_method = self.forward_xpu
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.mamba.ops.cpu.gdn_attention import (
                register_cpu_gdn_attention_ops,
            )

            register_cpu_gdn_attention_ops()
            self._forward_method = self.forward_cpu
        elif current_platform.is_rocm():
            self._forward_method = self.forward_hip
        else:
            self._forward_method = self.forward_cuda

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
        # we need to create qkvz_proj adaptively here.
        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),
        # in_proj_qkv and in_proj_z are created separately instead.
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )

        # ba_proj doesn't support blockwise fp8 quantization.
        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint
        # layouts, so we use a factory method to create the projection.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )
        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [
                query_key_settings,
                query_key_settings,
                value_settings,
            ],
            self.tp_size,
            self.tp_rank,
        )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        assert output_gate_type in ["silu", "swish", "sigmoid"], (
            f"unsupported {output_gate_type=}"
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_type,
            device=current_platform.current_device(),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=reduce_results,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        self.gdn_prefill_backend = self.chunk_gated_delta_rule.gdn_prefill_backend
        self._prefill_kernels_warmed_up = False
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )
        self.gdn_decode_kernel = envs.VLLM_GDN_DECODE_KERNEL.strip().lower()
        if self.gdn_decode_kernel == "cuda":
            reason = self._fused_gdn_decode_unsupported_reason(vllm_config)
            if reason is not None:
                if "VLLM_GDN_DECODE_KERNEL" in os.environ:
                    raise ValueError(
                        f"VLLM_GDN_DECODE_KERNEL=cuda is not supported: {reason}"
                    )
                logger.info_once(
                    "Falling back to the Triton GDN decode path: %s", reason
                )
                self.gdn_decode_kernel = "triton"
        self.enable_fused_gdn_decode = self.gdn_decode_kernel == "cuda"
        logger.info_once("GDN decode kernel: %s", self.gdn_decode_kernel)

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _fused_gdn_decode_unsupported_reason(
        self, vllm_config: VllmConfig
    ) -> str | None:
        conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()
        if (
            self.gqa_interleaved_layout
            or self.head_k_dim != 128
            or self.head_v_dim != 128
            or self.norm.activation != "silu"
            or vllm_config.model_config.dtype != torch.bfloat16
            or conv_state_dtype != torch.bfloat16
            or recurrent_state_dtype not in FUSED_GDN_STATE_DTYPES
            or not current_platform.has_device_capability(80)
        ):
            return (
                "the fused CUDA kernel requires a BF16 GDN model with "
                "K=V=128, SiLU gating, non-interleaved GQA layout, BF16 "
                "convolution cache, BF16 or FP32 recurrent state, and a "
                "GPU with compute capability 8.0+"
            )
        if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
            return "torch.ops._C.fused_gdn_decode_post_conv_mtp is not built"
        return None

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), qkvz weights are
        # stored as a single fused tensor with interleaved GQA layout, so we
        # use one output shard to preserve the interleaving across TP ranks.
        # When gqa_interleaved_layout=False (Qwen3.5), the checkpoint has
        # separate q, k, v, z weights, so we use 4 independent output sizes.
        output_sizes = (
            [sum((key_dim, key_dim, value_dim, value_dim))]
            if self.gqa_interleaved_layout
            else [key_dim, key_dim, value_dim, value_dim]
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), in_proj_ba is stored
        # as a single fused weight [b_g0, a_g0, b_g1, a_g1, ...] interleaved
        # by key-head group; a single output shard preserves this across TP.
        # When gqa_interleaved_layout=False (Qwen3.5), in_proj_b and in_proj_a
        # are separate checkpoint weights, so we use 2 independent output sizes.
        output_sizes = (
            [num_v_heads * 2] if self.gqa_interleaved_layout else [num_v_heads] * 2
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=self.maybe_disable_tp(quant_config),
        )

    def maybe_disable_tp(self, quant_config: QuantizationConfig | None) -> bool:
        """Whether to replicate ba_proj instead of TP-sharding it.

        Marlin requires output_size_per_partition >= MIN_THREAD_N=64, which
        the Qwen3.5 non-interleaved [num_v_heads]*2 layout violates at TP>=2
        (e.g. num_v_heads=64, TP=4 -> 16). Replicating the projection keeps
        each rank above the Marlin threshold; forward() then slices b/a to
        the local TP partition. Qwen3-Next's interleaved [num_v_heads*2]
        layout is unaffected and stays TP-sharded.

        See https://github.com/vllm-project/vllm/issues/35924
        """
        return (
            current_platform.is_cuda()
            and not self.gqa_interleaved_layout
            and isinstance(quant_config, (AutoAWQConfig, AutoGPTQConfig, INCConfig))
        )

    def split_ba(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, a = ba.chunk(2, dim=-1)
        if self.disable_tp_for_ba_proj and self.tp_size > 1:
            # ba_proj is replicated for Marlin; slice b/a to local TP rank.
            ba_chunk = self.num_v_heads // self.tp_size
            ba_start = self.tp_rank * ba_chunk
            b = b[:, ba_start : ba_start + ba_chunk]
            a = a[:, ba_start : ba_start + ba_chunk]
        return b, a

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    @torch.compile(fullgraph=True)
    def prepare_gdn_attention_core_inputs(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
        num_tokens: int,
    ):
        """
        Derives mixed_qkv, z, b, a from projected qkvz/ba for the GDN custom op.

        For gqa_interleaved_layout (Qwen3-Next): unpack the interleaved
        [ng, (hk + hk + np/ng*hv + np/ng*hv)] layout into contiguous qkv.
        For non-interleaved layout (Qwen3.5): simple split along last dim.
        """
        if not self.gqa_interleaved_layout:
            # Qwen3.5: weights are in [q, k, v, z] order
            assert num_tokens == mixed_qkvz.shape[0]
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z_flat = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            n = mixed_qkvz.shape[0]
            z_out = z_flat.reshape(n, -1, self.head_v_dim)
            b, a = mixed_ba.chunk(2, dim=-1)
            return mixed_qkv, z_out, b, a

        # Qwen3-Next: interleaved GQA layout
        base_shape_qkvz = mixed_qkvz.size()[:-1]
        base_shape_ba = mixed_ba.size()[:-1]
        ng = self.num_k_heads // self.tp_size

        new_tensor_shape_qkvz = base_shape_qkvz + (
            ng,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = base_shape_ba + (
            ng,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=-1)

        mixed_qkv_logical = torch.cat(
            [
                query.reshape(num_tokens, -1),
                key.reshape(num_tokens, -1),
                value.reshape(num_tokens, -1),
            ],
            dim=-1,
        )

        # The split above produces non-contiguous views into the interleaved
        # buffer.  Concatenating everything into a single flat tensor forces a
        # contiguous copy, then slicing back out gives contiguous q/k/v/z/b/a
        # tensors that downstream kernels require.  Doing this in one cat+slice
        # keeps torch.compile in a single Triton graph instead of emitting
        # separate copy kernels per tensor.  The original code used
        # rearrange(...).contiguous() on each tensor individually.
        fused = torch.cat(
            [
                mixed_qkv_logical.reshape(-1),
                z.reshape(-1),
                b.reshape(-1),
                a.reshape(-1),
            ],
            dim=0,
        )

        curr = 0
        qkv_numel = mixed_qkv_logical.numel()
        z_numel = z.numel()
        b_numel = b.numel()
        a_numel = a.numel()

        mixed_qkv_out = fused[curr : curr + qkv_numel].view(num_tokens, -1)
        curr += qkv_numel

        z_out = fused[curr : curr + z_numel].view(
            num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim
        )
        curr += z_numel

        b_out = fused[curr : curr + b_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )
        curr += b_numel

        a_out = fused[curr : curr + a_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )

        return mixed_qkv_out, z_out, b_out, a_out

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

        The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)``
        followed by ``.contiguous()`` on each tensor.  This version flattens
        all three splits into a single buffer via ``torch.cat`` so that
        torch.compile emits one Triton copy kernel instead of three separate
        contiguous() calls.
        """
        if mixed_qkv is None:
            return None, None, None

        seq_len = mixed_qkv.shape[0]
        q_dim = self.key_dim // self.tp_size
        k_dim = self.key_dim // self.tp_size
        v_dim = self.value_dim // self.tp_size

        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

        fused = torch.cat(
            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
        )

        q_size = seq_len * q_dim
        k_size = seq_len * k_dim

        q_contig = fused[0:q_size]
        k_contig = fused[q_size : q_size + k_size]
        v_contig = fused[q_size + k_size :]

        query = q_contig.view(1, seq_len, -1, self.head_k_dim)
        key = k_contig.view(1, seq_len, -1, self.head_k_dim)
        value = v_contig.view(1, seq_len, -1, self.head_v_dim)

        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_method(hidden_states)

    def _output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Part 3: RMSNormGated + output linear projection.

        The RMSNormGated + quant sequence is eligible for fusion
        by the compilation pass when fuse_norm_quant is enabled.
        """
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        output, _ = self.out_proj(core_attn_out)
        return output

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """ROCm forward using AITER Triton fused projection+attention when
        available, otherwise falling back to the generic CUDA path."""
        if GDN_AITER_TRITON_AVAILABLE:
            num_tokens = hidden_states.size(0)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            projected_states_qkvz = projected_states_qkvz.view(num_tokens, -1)
            projected_states_ba = projected_states_ba.view(num_tokens, -1)
            core_attn_out = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            z = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=projected_states_qkvz.dtype,
                device=projected_states_qkvz.device,
            )

            torch.ops.vllm.qwen_gdn_attention_core(
                projected_states_qkvz,
                projected_states_ba,
                z,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
                use_aiter=True,
            )

            return self._output_projection(core_attn_out, z)
        else:
            return self.forward_cuda(hidden_states)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        use_fused_gdn_decode = (
            self.enable_fused_gdn_decode
            and hidden_states.dtype == torch.bfloat16
            and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
        )
        if use_fused_gdn_decode:
            core_attn_out = torch.zeros(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                mixed_qkvz,
                ba,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
            )
            output, _ = self.out_proj(core_attn_out.flatten(-2))
            return output

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self.split_ba(ba)

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b.contiguous(),
            a.contiguous(),
            core_attn_out,
            layer_name=_encode_layer_name(self.prefix),
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        return self._output_projection(core_attn_out, z)

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        # ============================================================
        # Part 2: Core Attention
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)

        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert not hasattr(self, "in_proj_qkv"), "lora isn't supported on CPU."

        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)

        num_tokens = hidden_states.size(0)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.cpu_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        """Warm up GDN prefill kernels during V1 profiling.

        During V1 profile runs, ``_forward_core`` returns early because
        ``attn_metadata`` is ``None``, so the autotuned kernels used by
        ``chunk_gated_delta_rule`` (e.g. ``solve_tril``,
        ``chunk_scaled_dot_kkt``) are never invoked.  After profiling,
        vLLM allocates KV cache using most of the remaining GPU memory.
        When the first real inference triggers the autotuner it OOMs
        because there is not enough memory left for benchmarking.

        This method runs minimal forward passes through
        ``chunk_gated_delta_rule`` with small dummy tensors to force
        autotuning while GPU memory is still plentiful.  The autotuner
        results are cached globally, so only the first layer incurs
        actual benchmarking cost.

        All kernels including ``chunk_fwd_kernel_o`` now use a fixed
        ``BT = chunk_size`` (64).  A single warmup pass with T = 64
        is sufficient to populate the autotuner cache.

        The decode path uses ``gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule``
        which has fixed kernel parameters (no autotuning), so only the
        prefill (chunked) path needs warming up.
        """
        if self._prefill_kernels_warmed_up:
            return
        self._prefill_kernels_warmed_up = True

        device = qkv_or_qkvz.device
        dtype = qkv_or_qkvz.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()

        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Mirror the real
        # prefill path here: build q/k/v/g/beta via fused_post_conv_prep and
        # then run chunk_gated_delta_rule with in-kernel L2 norm disabled.
        T = FLA_CHUNK_SIZE
        dummy_mixed_qkv = torch.randn(
            T, qkv_or_qkvz.shape[-1] - v_dim, device=device, dtype=dtype
        )
        dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=dummy_mixed_qkv,
            a=dummy_a,
            b=dummy_b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)
        state = torch.zeros(
            1,
            num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

        # CuteDSL kernels require metadata
        chunk_indices = None
        chunk_offsets = None
        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, T)

        try:
            self.chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
        except Exception:
            logger.warning(
                "GDN prefill kernel warmup (T=%d) failed for "
                "layer %s. First inference may OOM due to "
                "autotuner.",
                T,
                self.prefix,
                exc_info=True,
            )
        else:
            logger.debug(
                "GDN prefill kernel warmup (T=%d) completed for layer %s",
                T,
                self.prefix,
            )
        finally:
            del (
                dummy_mixed_qkv,
                q,
                k,
                v,
                dummy_a,
                dummy_b,
                g,
                beta,
                state,
                cu_seqlens,
                chunk_indices,
                chunk_offsets,
            )

        torch.accelerator.empty_cache()

    def _forward_core_rocm(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """ROCm AITER fast path: conv1d + recurrent attention from packed
        qkvz/ba layout.

        For decode-only (no spec, no prefill) interleaved-GQA layouts,
        dispatches directly to ``_forward_core_decode_aiter``. Otherwise unpacks
        the packed layout and falls through to ``_forward_core``.

        Args:
            qkvz: packed [q, k, v, z] projection (num_tokens, qkvz_dim)
            ba:   packed [b, a] gating vectors    (num_tokens, 2*num_heads)
            z_out: **output** buffer for z        (num_tokens, num_heads,
                   head_dim); mutated in-place.
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            v_dim = core_attn_out.shape[-1] * core_attn_out.shape[-2]
            self._warmup_prefill_kernels(qkvz, v_dim)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # The AITER fused reshape/conv kernel expects Qwen3-Next's interleaved
        # GQA layout. Qwen3.5 uses a non-interleaved q/k/v/z layout and must use
        # the generic path below to split/rearrange inputs correctly.
        if (
            self.gqa_interleaved_layout
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_aiter(
                qkvz=qkvz,
                ba=ba,
                z_out=z_out,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        core_attn_out.zero_()
        num_tokens_all = qkvz.shape[0]
        mixed_qkv, z, b, a = self.prepare_gdn_attention_core_inputs(
            qkvz, ba, num_tokens_all
        )
        z_out[:] = z
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
        )

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core conv1d + recurrent attention (standard path).

        Args:
            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
            b: beta gating vector                   (num_tokens, num_heads)
            a: alpha gating vector                  (num_tokens, num_heads)
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if (
            self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                a_spec = a
                b_spec = b
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                a_spec = a.index_select(0, spec_token_indx)
                b_spec = b.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            _info = None
            if _ddtree_active():
                from vllm.v1.spec_decode.ddtree.gdn import (register_state,
                                                            tolist_cached)

                _n_spec = attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                # 🔴 계층마다 내리면 스텝당 24회 D2H 다. 내용은 계층 간 동일하다.
                _cl = tolist_cached("qsl", spec_query_start_loc[: _n_spec + 1])
                _T = [_cl[i + 1] - _cl[i] for i in range(_n_spec)]
                _info = _ddtree_gdn(_n_spec, _T, conv_weights.shape[-1])
                if _info is not None:
                    register_state(
                        id(self), conv_state, ssm_state,
                        tolist_cached("ssi",
                                      spec_state_indices_tensor[:_n_spec]),
                        conv_weights.shape[-1], _info["keys"],
                    )
            self._ddtree_info = _info
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                # DDTree: 있으면 커널이 윈도를 조상 열에서 읽는다
                tree_cols=None if _info is None else _info["tree_cols"],
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

        # Split mixed non-spec-decode+prefill to process independently
        split_non_spec = (
            spec_sequence_masks is None
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            if split_non_spec:
                conv_output_prefill = mixed_qkv_non_spec[num_decode_tokens:]
                a_prefill = a_non_spec[num_decode_tokens:]
                b_prefill = b_non_spec[num_decode_tokens:]
            else:
                conv_output_prefill = mixed_qkv_non_spec
                a_prefill = a_non_spec
                b_prefill = b_non_spec

            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = fused_post_conv_prep(
                conv_output=conv_output_prefill,
                a=a_prefill,
                b=b_prefill,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=self.num_k_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec = None
            beta_non_spec = None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a_spec,
                    b=b_spec,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_spec_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    # DDTree: conv 와 '같은' _info 를 써서 둘이 어긋나지 않게 한다
                    tree_parent_indices=(
                        self._ddtree_info["parents"]
                        if getattr(self, "_ddtree_info", None) is not None
                        else None
                    ),
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process non-spec-decode part
        if split_non_spec:
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec[:num_decode_tokens]  # type: ignore[index]
            )
            core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a[:num_decode_tokens],
                b=b[:num_decode_tokens],
                dt_bias=self.dt_bias,
                q=query_decode,
                k=key_decode,
                v=value_decode,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_decode = None

        # 2.3: Process the remaining part (prefill chunk, or non-spec decode-only)
        if attn_metadata.num_prefills > 0:
            # State indices, initial-state mask and cu_seqlens for the chunk
            # kernel are precomputed by the metadata builder (the prefill tail
            # when decodes are peeled off, else the full non-spec batch), so they
            # don't need to be re-derived per layer.
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            initial_state = ssm_state[prefill_state_indices]
            initial_state[~prefill_has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=attn_metadata.prefill_query_start_loc,
                chunk_indices=attn_metadata.chunk_indices,
                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
            # Init cache
            ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)

            if split_non_spec:
                # Stitch the peeled decode outputs in front of the prefill
                # outputs (decode-first order).
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_non_spec], dim=1
                )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def _forward_core_decode_aiter(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_non_spec, b, a = (
            gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
                qkvz,
                attn_metadata.num_actual_tokens,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
                ba,
                z_out,
                core_attn_out,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        )

        # 2. Recurrent attention
        gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            qkv=mixed_qkv_non_spec,
            key_dim=self.key_dim // self.tp_size,
            value_dim=self.value_dim // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],  # type: ignore[index]
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
            core_attn_out=core_attn_out.reshape(-1),
        )

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """
        Core attention computation with a packed non-spec decode fast path.
        """
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            validate_data=False,
        )
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            use_qk_l2norm_in_kernel=True,
        )
        return

    def _forward_core_decode_spec_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        state_indices = attn_metadata.spec_state_indices_tensor
        cu_seqlens = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert state_indices is not None
        assert cu_seqlens is not None
        assert num_accepted_tokens is not None

        num_requests = attn_metadata.num_spec_decodes
        num_actual_tokens = attn_metadata.num_actual_tokens
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        # --- DDTree: 융합 경로에서도 트리 정보를 세운다 (conv + SSM 이 같은 _info) ---
        # 🔴 요청마다 드래프트 길이가 다를 수 있다. cu_seqlens 에서 실제 길이를
        #    뽑는다 (균등 분할 가정을 쓰면 가변 길이에서 어긋난다).
        _info = None
        if _ddtree_active():
            from vllm.v1.spec_decode.ddtree.gdn import (register_state,
                                                        tolist_cached)

            _cl = tolist_cached("cus", cu_seqlens[: num_requests + 1])
            _T = [_cl[i + 1] - _cl[i] for i in range(num_requests)]
            _info = _ddtree_gdn(num_requests, _T, conv_weights.shape[-1])
            if _info is not None:
                register_state(
                    id(self), conv_state, self.kv_cache[1],
                    tolist_cached("si", state_indices[:num_requests]),
                    conv_weights.shape[-1], _info["keys"],
                )
        self._ddtree_info = _info
        # --- DDTree 진단: conv 상태를 스텝 경계로 덤프한다 (VLLM_DDTREE_CONVDUMP=N) ---
        #     합성 호출이 아니라 프로덕션 인자 그대로 잡아야 인계 검증이 성립한다.
        _cd = _ddtree_conv_dump_begin(
            self, conv_state, mixed_qkv[:num_actual_tokens], state_indices,
            num_accepted_tokens, num_requests,
            None if _info is None else _info["tree_cols"])
        if _info is not None:
            from vllm.v1.spec_decode.ddtree.gdn import validate_tree_inputs
            validate_tree_inputs(
                parents=_info["parents"], cu_seqlens=cu_seqlens,
                state_indices=state_indices, tree_cols=_info.get("tree_cols"),
                conv_state=conv_state, n_reqs=num_requests, tag="융합 conv")
        mixed_qkv = causal_conv1d_update(
            mixed_qkv[:num_actual_tokens],
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=state_indices[:num_requests, 0],
            num_accepted_tokens=num_accepted_tokens[:num_requests],
            tree_cols=None if _info is None else _info["tree_cols"],
            query_start_loc=cu_seqlens[: num_requests + 1],
            max_query_len=state_indices.size(1),
            validate_data=False,
        )
        _ddtree_conv_dump_end(_cd, conv_state)
        self._forward_core_decode_spec_post_conv_fused_norm(
            mixed_qkv=mixed_qkv,
            b=b[:num_actual_tokens],
            a=a[:num_actual_tokens],
            output_gate=output_gate[:num_actual_tokens],
            core_attn_out=core_attn_out[:num_actual_tokens],
            attn_metadata=attn_metadata,
        )

    def _forward_core_decode_spec_post_conv_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        state_indices = attn_metadata.spec_state_indices_tensor
        cu_seqlens = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert state_indices is not None
        assert cu_seqlens is not None
        assert num_accepted_tokens is not None

        num_requests = attn_metadata.num_spec_decodes

        # --- DDTree: 트리면 8토큰 상한 없는 우리 CUDA 커널로 ---
        _info = getattr(self, "_ddtree_info", None)
        # 🔴 트리 정보가 없어도, 폭이 원본 커널의 8토큰 상한을 넘으면 원본으로
        #    물러설 수 없다 (state_indices must have shape [N,S] with S<=8).
        #    이때는 사슬 부모를 만들어 우리 커널로 처리한다 — 사슬은 원본과
        #    의미가 같고 우리 커널에는 상한이 없다. 배치에 트리 있는 요청과
        #    없는 요청이 섞이면 실제로 여기로 온다.
        if _info is None and state_indices.size(1) > MAX_FUSED_GDN_MTP_TOKENS:
            _cl = cu_seqlens[: num_requests + 1].tolist()
            _mx = state_indices.size(1)
            _rows = []
            for _r in range(num_requests):
                _L = _cl[_r + 1] - _cl[_r]
                _rows.append([-1] + list(range(_L - 1)) + [-1] * (_mx - _L))
            _info = {"parents": torch.tensor(_rows, dtype=torch.int32,
                                             device=state_indices.device)}
        if _info is not None:
            from vllm.v1.spec_decode.ddtree.cuda_ext import get_ext

            _ext = get_ext()
            if _ext is not None:
                import os as _os
                _chk = _os.environ.get("VLLM_DDTREE_GDN_CHECK") in ("1", "2")
                if _chk:
                    # 🔴 상태 캐시 전체는 GB 단위다 — 이번 스텝이 건드리는 슬롯만 복제한다
                    _slots = state_indices[:num_requests].reshape(-1).unique()
                    _snap = self.kv_cache[1][_slots].clone()
                from vllm.v1.spec_decode.ddtree.gdn import validate_tree_inputs
                validate_tree_inputs(
                    parents=_info["parents"], cu_seqlens=cu_seqlens,
                    state_indices=state_indices, ssm_state=self.kv_cache[1],
                    n_reqs=num_requests, tag="융합 SSM")
                _ext.gdn_decode_tree_mtp(
                    mixed_qkv, a, b, self.A_log, self.dt_bias,
                    state_indices[:num_requests].contiguous(),
                    cu_seqlens[: num_requests + 1].contiguous(),
                    num_accepted_tokens[:num_requests].contiguous(),
                    _info["parents"].contiguous(),
                    self.kv_cache[1], output_gate, self.norm.weight,
                    core_attn_out, self.head_k_dim**-0.5,
                    self.layer_norm_epsilon,
                )
                if _chk:
                    _ddtree_gdn_crosscheck(
                        self, _slots, _snap, mixed_qkv, a, b, state_indices,
                        cu_seqlens, num_accepted_tokens, _info, num_requests,
                        core_attn_out, output_gate)
                return

        ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state_indices=state_indices[:num_requests],
            cu_seqlens=cu_seqlens[: num_requests + 1],
            num_accepted_tokens=num_accepted_tokens[:num_requests],
            state=self.kv_cache[1],
            output_gate=output_gate,
            norm_weight=self.norm.weight,
            out=core_attn_out,
            scale=self.head_k_dim**-0.5,
            norm_eps=self.layer_norm_epsilon,
        )

    def _forward_core_fused_norm_packed(
        self,
        mixed_qkvz: torch.Tensor,
        ba: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkvz[:, :qkv_size], 0)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)
        mixed_qkv, output_gate_flat = mixed_qkvz.split(
            [qkv_size, self.value_dim // self.tp_size], dim=-1
        )
        output_gate = output_gate_flat.reshape(
            output_gate_flat.size(0), -1, self.head_v_dim
        )
        b, a = self.split_ba(ba)
        self._forward_core_fused_norm(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            output_gate=output_gate,
            core_attn_out=core_attn_out,
        )

    def _can_use_fused_gdn_mtp_decode(
        self, attn_metadata: GDNAttentionMetadata
    ) -> bool:
        state_indices = attn_metadata.spec_state_indices_tensor
        return (
            attn_metadata.spec_sequence_masks is not None
            and attn_metadata.num_decodes == 0
            and attn_metadata.num_spec_decodes > 0
            and self.kv_cache[1].dtype in FUSED_GDN_STATE_DTYPES
            and self.gdn_decode_kernel == "cuda"
            and self.num_v_heads % self.num_k_heads == 0
            and self.num_v_heads // self.num_k_heads in (1, 2, 3, 4, 8)
            and state_indices is not None
            # 🔴 DDTree: 우리 커널은 8토큰 상한이 없다 (공유메모리에서 shared_state 를
            #    제거하고 토큰 축을 타일링했다). 트리 공급자가 등록돼 있으면 상한을 푼다.
            #    원본 커널은 여전히 8 초과를 거부하므로, 트리가 아닐 때는 그대로 둔다.
            # 🔴 상한을 푸는 조건은 "프로바이더 등록" 이 아니라 "우리 커널을 실제로
            #    쓸 수 있는가" 다. 커널이 없으면 원본으로 떨어지는데 원본은 8토큰
            #    상한이 있어 죽는다 (state_indices must have shape [N,S] S<=8).
            #    커널이 없으면 융합 경로를 아예 쓰지 않고 Triton 으로 간다.
            and (
                state_indices.size(1) <= MAX_FUSED_GDN_MTP_TOKENS
                or (_DDTREE_GDN_PROVIDER is not None and _ddtree_kernel_ready())
            )
            and hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")
        )

    def _rms_norm_gated_cuda(
        self,
        x: torch.Tensor,
        output_gate: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
            layer_norm_fwd,
        )

        x_shape = x.shape
        assert output_gate.shape == x_shape
        assert out.shape == x_shape
        x_2d = x.reshape(-1, x_shape[-1])
        output_gate_2d = output_gate.reshape(-1, x_shape[-1])
        out_2d = out.reshape(-1, x_shape[-1])
        assert x_2d.stride(-1) == 1
        assert output_gate_2d.stride(-1) == 1
        assert out_2d.stride(-1) == 1
        layer_norm_fwd(
            x_2d,
            self.norm.weight.contiguous(),
            self.norm.bias,
            self.norm.eps,
            z=output_gate_2d,
            out=out_2d,
            group_size=(
                x_shape[-1] if self.norm.group_size is None else self.norm.group_size
            ),
            norm_before_gate=self.norm.norm_before_gate,
            is_rms_norm=True,
            activation=self.norm.activation,
        )

    def _forward_core_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)
        if (
            self._can_use_fused_gdn_mtp_decode(attn_metadata)
            and attn_metadata.num_prefills == 0
        ):
            self._forward_core_decode_spec_fused_norm(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                output_gate=output_gate,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )
            return
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b.contiguous(),
            a=a.contiguous(),
            core_attn_out=core_attn_out,
        )
        num_actual_tokens = attn_metadata.num_actual_tokens
        self._rms_norm_gated_cuda(
            core_attn_out[:num_actual_tokens],
            output_gate[:num_actual_tokens],
            core_attn_out[:num_actual_tokens],
        )


def qwen_gdn_attention_core(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    use_aiter: bool = False,
) -> None:
    """Custom op dispatching to _forward_core or _forward_core_rocm.

    Handles conv1d + recurrent attention only; input/output projections
    are performed by the caller.

    When ``use_aiter=False`` (standard path):
        qkv_or_qkvz is [q, k, v], b_or_ba is b, a_or_z_out is a (read-only).
    When ``use_aiter=True`` (AITER Triton path, ROCm only):
        qkv_or_qkvz is [q, k, v, z], b_or_ba is [b, a], a_or_z_out is the
        z output buffer (mutated in-place).

    ``core_attn_out`` is always mutated in-place.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if use_aiter:
        self._forward_core_rocm(
            qkvz=qkv_or_qkvz,
            ba=b_or_ba,
            z_out=a_or_z_out,
            core_attn_out=core_attn_out,
        )
    else:
        self._forward_core(
            mixed_qkv=qkv_or_qkvz,
            b=b_or_ba,
            a=a_or_z_out,
            core_attn_out=core_attn_out,
        )


def gdn_attention_core_fake(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    use_aiter: bool = False,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="qwen_gdn_attention_core",
    op_func=qwen_gdn_attention_core,
    mutates_args=["a_or_z_out", "core_attn_out"],
    fake_impl=gdn_attention_core_fake,
)


def qwen_gdn_attention_core_fused_norm_packed(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core_fused_norm_packed(
        mixed_qkvz=mixed_qkvz,
        ba=ba,
        core_attn_out=core_attn_out,
    )


def gdn_attention_core_fused_norm_packed_fake(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen_gdn_attention_core_fused_norm_packed",
    op_func=qwen_gdn_attention_core_fused_norm_packed,
    mutates_args=["core_attn_out"],
    fake_impl=gdn_attention_core_fused_norm_packed_fake,
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output


# ---------------------------------------------------------------------------
# DDTree 진단: 같은 입력에 CUDA 트리 커널과 Triton 참조를 모두 돌려 SSM 상태를 비교한다.
# VLLM_DDTREE_GDN_CHECK=1 일 때만 동작한다. 상태 캐시는 GB 단위라 전체를 복제하면
# OOM 이 나므로, 이번 스텝이 건드리는 슬롯만 스냅샷한다.
# ---------------------------------------------------------------------------
_DDTREE_CHECK_STATE = {"calls": 0, "bad": 0, "dumped": False}


def _ddtree_gdn_crosscheck(layer, slots, snap, mixed_qkv, a, b, state_indices,
                           cu_seqlens, num_accepted_tokens, info, num_requests,
                           core_attn_out, output_gate):
    import os
    import torch
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )
    d = _DDTREE_CHECK_STATE
    d["calls"] += 1
    st_cuda = layer.kv_cache[1][slots].clone()
    out_cuda = core_attn_out.clone()
    layer.kv_cache[1][slots] = snap              # Triton 을 같은 초기상태에서 출발시킨다
    q, k, v = layer.rearrange_mixed_qkv(mixed_qkv)
    out_tri_raw, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=layer.A_log, a=a, b=b, dt_bias=layer.dt_bias, q=q, k=k, v=v,
        initial_state=layer.kv_cache[1], inplace_final_state=True,
        cu_seqlens=cu_seqlens[: num_requests + 1],
        ssm_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
        tree_parent_indices=info["parents"],
        use_qk_l2norm_in_kernel=True,
    )
    st_tri = layer.kv_cache[1][slots].clone()
    # CUDA 커널은 게이팅+RMSNorm 을 내부에서 한다 — Triton 쪽에도 같은 후처리를 건다
    # Triton 쪽은 앞에 배치 축이 붙어 나올 수 있다 — CUDA 출력 모양으로 맞춘다
    out_tri_raw = out_tri_raw.reshape(out_cuda.shape).contiguous()
    out_tri = torch.empty_like(out_cuda)
    try:
        layer._rms_norm_gated_cuda(out_tri_raw, output_gate, out_tri)
    except Exception as e:                                   # 진단이 본 실행을 죽이지 않게
        if not d.get("normwarn"):
            d["normwarn"] = True
            print(f"[DDTREE-GDN 교차검증] 정규화 건너뜀: {type(e).__name__} {e} "
                  f"out={tuple(out_cuda.shape)} gate={tuple(output_gate.shape)}", flush=True)
        out_tri = out_tri_raw

    def _rel(x, y):
        den = max(1e-6, y.float().abs().max().item())
        return (x.float() - y.float()).abs().max().item() / den

    # 🔴 깊은 트리 스텝을 골라 덤프한다. fp64 검증을 얕은 스텝에만 하면
    #    "커널 정상" 이 깊이 8 에서도 성립한다는 보장이 없다.
    _want = int(os.environ.get("VLLM_DDTREE_DUMP_DEPTH", "0"))
    if _want and not d.get("dumped_deep"):
        _p = info["parents"][0].tolist()
        _dep = [0] * len(_p)
        for _i in range(1, len(_p)):
            _dep[_i] = _dep[_p[_i]] + 1 if _p[_i] >= 0 else 0
        if max(_dep) >= _want:
            d["dumped_deep"] = True
            print(f"[DDTREE-GDN] 깊이 {max(_dep)} 스텝 덤프 (호출#{d['calls']})", flush=True)
            torch.save({"mixed_qkv": mixed_qkv.cpu(), "a": a.cpu(), "b": b.cpu(),
                        "state_indices": state_indices.cpu(), "cu_seqlens": cu_seqlens.cpu(),
                        "num_accepted": num_accepted_tokens.cpu(),
                        "parents": info["parents"].cpu(), "slots": slots.cpu(),
                        "snap": snap.cpu(), "output_gate": output_gate.cpu(),
                        "A_log": layer.A_log.cpu(), "dt_bias": layer.dt_bias.cpu(),
                        "norm_weight": layer.norm.weight.cpu(),
                        "num_requests": num_requests, "max_depth": max(_dep)},
                       os.environ.get("VLLM_DDTREE_DEEPDUMP", "/work/gdn_deep.pt"))
    r_st, r_out = _rel(st_cuda, st_tri), _rel(out_cuda, out_tri)
    d["max_state"] = max(d.get("max_state", 0.0), r_st)
    d["max_out"] = max(d.get("max_out", 0.0), r_out)
    if d["calls"] % 400 == 0:
        print(f"[DDTREE-GDN 교차검증] {d['calls']}회  누적 최대 상대오차 "
              f"상태 {d['max_state']:.5f} / 출력 {d['max_out']:.5f}", flush=True)
    dif, ref = max(r_st, r_out), 1.0
    if dif > 1e-3:
        d["bad"] += 1
        if not d["dumped"]:
            d["dumped"] = True
            print(f"[DDTREE-GDN 불일치] 호출#{d['calls']} 상대오차 상태 {r_st:.5f} "
                  f"출력 {r_out:.5f} T={mixed_qkv.shape[0]//max(1,num_requests)} "
                  f"accepted={num_accepted_tokens[:num_requests].tolist()} "
                  f"부모={info['parents'][:num_requests].tolist()}", flush=True)
            torch.save({"mixed_qkv": mixed_qkv.cpu(), "a": a.cpu(), "b": b.cpu(),
                        "state_indices": state_indices.cpu(),
                        "cu_seqlens": cu_seqlens.cpu(),
                        "num_accepted": num_accepted_tokens.cpu(),
                        "parents": info["parents"].cpu(),
                        "slots": slots.cpu(), "snap": snap.cpu(),
                        "output_gate": output_gate.cpu(),
                        "A_log": layer.A_log.cpu(), "dt_bias": layer.dt_bias.cpu(),
                        "norm_weight": layer.norm.weight.cpu(),
                        "num_requests": num_requests},
                       os.environ.get("VLLM_DDTREE_DUMP", "/work/gdn_mismatch.pt"))
    # VLLM_DDTREE_GDN_CHECK=2 : 실행을 Triton 결과로 이어간다 (A/B 확정용).
    # 출력만 바꿔치기해서 무손실이 회복되면 0.2% 출력 차이가 원인이고,
    # 그대로 깨지면 교차검증이 '비교하지 않는' 지점(융합 경로 conv 등)이 원인이다.
    if os.environ.get("VLLM_DDTREE_GDN_CHECK") == "2":
        core_attn_out.copy_(out_tri)
        return                                    # 상태도 Triton 것을 남긴다
    layer.kv_cache[1][slots] = st_cuda           # 실행은 CUDA 결과로 계속한다


# ---------------------------------------------------------------------------
# DDTree 진단: conv 상태 스텝 경계 덤프
#   스텝 N 의 conv 호출 전/후 블록과, 스텝 N+1 의 호출 전 블록(= 압축 반영 후)을
#   모은다. 이걸로 "압축이 남긴 상태" 와 "다음 스텝이 실제로 읽는 열" 을 대조한다.
# ---------------------------------------------------------------------------
_CONV_DUMP = {"want": 0, "layer": None, "recs": [], "path": None}


def _ddtree_conv_dump_begin(layer, conv_state, x, state_indices,
                            num_accepted_tokens, num_requests, tree_cols):
    import os
    d = _CONV_DUMP
    if not d["want"]:
        d["want"] = int(os.environ.get("VLLM_DDTREE_CONVDUMP", "0") or 0)
        d["path"] = os.environ.get("VLLM_DDTREE_CONVDUMP_FILE", "/work/conv_steps.pt")
        if not d["want"]:
            d["want"] = -1
    if d["want"] < 0 or len(d["recs"]) >= d["want"]:
        return None
    if d["layer"] is None:
        d["layer"] = id(layer)          # 첫 GDN 계층만 추적한다 (용량)
    if id(layer) != d["layer"]:
        return None
    slot = int(state_indices[0, 0])
    return {"before": conv_state[slot].detach().clone().cpu(),
            "x": x.detach().clone().cpu(),
            "slot": slot,
            "state_indices": state_indices[:num_requests].detach().cpu(),
            "num_accepted": num_accepted_tokens[:num_requests].detach().cpu(),
            "tree_cols": None if tree_cols is None else tree_cols.detach().cpu()}


def _ddtree_conv_dump_end(rec, conv_state):
    import torch
    d = _CONV_DUMP
    if rec is None:
        return
    rec["after"] = conv_state[rec["slot"]].detach().clone().cpu()
    d["recs"].append(rec)
    if len(d["recs"]) >= d["want"]:
        torch.save(d["recs"], d["path"])
        print(f"[DDTREE-CONV] {len(d['recs'])} 스텝 덤프 → {d['path']}", flush=True)
