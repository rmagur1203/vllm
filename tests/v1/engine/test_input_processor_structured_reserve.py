# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`InputProcessor._maybe_reserve_tokens_for_structured_answer`.

With a reasoning parser the structured-output grammar is only enforced after
the reasoning block ends. Without a thinking budget, reasoning can eat the
whole `max_tokens` and the structured answer is never emitted. The processor
derives a `thinking_token_budget` for such requests; these tests pin down
when it does and does not.
"""

from types import SimpleNamespace

import pytest

from vllm import SamplingParams
from vllm.config import StructuredOutputsConfig
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.skip_global_cleanup


def _processor(
    *,
    reasoning_enabled: bool = True,
    reserve_ratio: float = 0.25,
    enable_in_reasoning: bool = False,
) -> SimpleNamespace:
    reasoning_config = SimpleNamespace(enabled=reasoning_enabled)
    return SimpleNamespace(
        vllm_config=SimpleNamespace(reasoning_config=reasoning_config),
        structured_outputs_config=StructuredOutputsConfig(
            reasoning_reserve_ratio=reserve_ratio,
            enable_in_reasoning=enable_in_reasoning,
        ),
    )


def _reserve(processor: SimpleNamespace, params: SamplingParams) -> None:
    InputProcessor._maybe_reserve_tokens_for_structured_answer(processor, params)


def _json_params(**kwargs) -> SamplingParams:
    return SamplingParams(
        structured_outputs=StructuredOutputsParams(json={"type": "object"}),
        **kwargs,
    )


def test_derives_budget_leaving_reserve_for_answer():
    params = _json_params(max_tokens=512)
    _reserve(_processor(), params)
    # 25% of 512 = 128 reserved for the answer; reasoning capped at 384.
    assert params.thinking_token_budget == 384


@pytest.mark.parametrize("max_tokens,expected", [(4, 3), (1, 0), (8000, 6000)])
def test_reserve_is_at_least_one_token(max_tokens, expected):
    params = _json_params(max_tokens=max_tokens)
    _reserve(_processor(), params)
    assert params.thinking_token_budget == expected


def test_explicit_request_budget_is_kept():
    params = _json_params(max_tokens=512, thinking_token_budget=100)
    _reserve(_processor(), params)
    assert params.thinking_token_budget == 100


def test_no_structured_output_untouched():
    params = SamplingParams(max_tokens=512)
    _reserve(_processor(), params)
    assert params.thinking_token_budget is None


def test_no_reasoning_untouched():
    # Without a reasoning parser the grammar applies from the first token,
    # so there is nothing to reserve.
    params = _json_params(max_tokens=512)
    _reserve(_processor(reasoning_enabled=False), params)
    assert params.thinking_token_budget is None


def test_reasoning_config_absent_untouched():
    params = _json_params(max_tokens=512)
    processor = SimpleNamespace(
        vllm_config=SimpleNamespace(reasoning_config=None),
        structured_outputs_config=StructuredOutputsConfig(),
    )
    _reserve(processor, params)
    assert params.thinking_token_budget is None


def test_enable_in_reasoning_untouched():
    # Grammar already applies during reasoning; a budget would only cut it.
    params = _json_params(max_tokens=512)
    _reserve(_processor(enable_in_reasoning=True), params)
    assert params.thinking_token_budget is None


def test_ratio_zero_disables():
    params = _json_params(max_tokens=512)
    _reserve(_processor(reserve_ratio=0.0), params)
    assert params.thinking_token_budget is None


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_config_rejects_out_of_range_ratio(bad):
    with pytest.raises(ValueError, match="reasoning_reserve_ratio"):
        StructuredOutputsConfig(reasoning_reserve_ratio=bad)
