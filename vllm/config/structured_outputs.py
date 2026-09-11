# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import model_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

StructuredOutputsBackend = Literal[
    "auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"
]


@config
class StructuredOutputsConfig:
    """Dataclass which contains structured outputs config for the engine."""

    backend: StructuredOutputsBackend = "auto"
    """Which engine will be used for structured outputs (e.g. JSON schema,
    regex, etc) by default. With "auto", we will make opinionated choices
    based on request contents and what the backend libraries currently support,
    so the behavior is subject to change in each release."""
    disable_any_whitespace: bool = False
    """If `True`, json output will always be compact without any whitespace.
    If `False`, the model may generate whitespace between JSON fields,
    which is still valid JSON. This is only supported for xgrammar
    and guidance backends."""
    disable_additional_properties: bool = False
    """If `True`, the `guidance` backend will not use `additionalProperties`
    in the JSON schema. This is only supported for the `guidance` backend and
    is used to better align its behaviour with `outlines` and `xgrammar`."""
    reasoning_parser: str = ""
    """Select the reasoning parser depending on the model that you're using.
    This is used to parse the reasoning content into OpenAI API format."""
    reasoning_parser_plugin: str = ""
    """Path to a dynamically reasoning parser plugin that can be dynamically
    loaded and registered."""
    enable_in_reasoning: bool = False
    """Whether to use structured input for reasoning."""
    reasoning_reserve_ratio: float = 0.25
    """Fraction of `max_tokens` kept free for the structured answer when a
    reasoning parser is active.

    With a reasoning parser, the grammar is only enforced *after* the
    reasoning block ends. If the model's reasoning consumes the whole
    `max_tokens` budget, the request finishes with `finish_reason="length"`
    and an empty `content` — the structured answer is never produced. To
    prevent that, requests that use structured outputs and do not set
    `thinking_token_budget` themselves get one derived as
    `max_tokens - max(1, int(max_tokens * reasoning_reserve_ratio))`, so the
    reasoning is force-closed with room left for the answer.

    Set to 0 to disable the derived budget. Ignored when
    `enable_in_reasoning` is set (the grammar then applies during reasoning
    and nothing needs reserving)."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @model_validator(mode="after")
    def _validate_structured_output_config(self) -> Self:
        if self.disable_any_whitespace and self.backend not in ("xgrammar", "guidance"):
            raise ValueError(
                "disable_any_whitespace is only supported for "
                "xgrammar and guidance backends."
            )
        if self.disable_additional_properties and self.backend != "guidance":
            raise ValueError(
                "disable_additional_properties is only supported "
                "for the guidance backend."
            )
        if not 0.0 <= self.reasoning_reserve_ratio < 1.0:
            raise ValueError(
                "reasoning_reserve_ratio must be in [0, 1); got "
                f"{self.reasoning_reserve_ratio}."
            )
        return self
