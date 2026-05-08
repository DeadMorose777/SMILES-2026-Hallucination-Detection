"""
aggregation.py - Token aggregation strategy and feature extraction.

The official pipeline passes only hidden states and an attention mask. To build
response-only features, this module reconstructs prompt lengths by tokenizing
the prompts from the train and test CSV files in the same order as solution.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
AGGREGATION_MODE = "max_l12_l13"

_PROMPT_LENS: list[int] | None = None
_AGGREGATE_COUNTER = 0


def _load_prompt_lengths() -> list[int]:
    """Tokenize prompts once in solution.py call order: dataset, then test."""
    global _PROMPT_LENS
    if _PROMPT_LENS is not None:
        return _PROMPT_LENS

    repo_root = Path(__file__).resolve().parent
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompt_lens: list[int] = []
    for file_name in ("dataset.csv", "test.csv"):
        path = repo_root / "data" / file_name
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["prompt"])
        prompt_lens.extend(
            len(tokenizer(str(prompt), add_special_tokens=False)["input_ids"])
            for prompt in df["prompt"]
        )

    _PROMPT_LENS = prompt_lens
    return prompt_lens


def _next_prompt_length() -> int | None:
    global _AGGREGATE_COUNTER
    prompt_lens = _load_prompt_lengths()
    prompt_len = None
    if _AGGREGATE_COUNTER < len(prompt_lens):
        prompt_len = prompt_lens[_AGGREGATE_COUNTER]
    _AGGREGATE_COUNTER += 1
    return prompt_len


def _safe_layer(hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
    idx = min(layer_idx, hidden_states.shape[0] - 1)
    return hidden_states[idx]


def _max_pool(layer: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return layer.index_select(0, positions).max(dim=0).values


def _mean_pool(layer: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return layer.index_select(0, positions).mean(dim=0)


def _fallback_feature(hidden_states: torch.Tensor, token_pos: int | torch.Tensor) -> torch.Tensor:
    layer_13 = _safe_layer(hidden_states, 13)
    if AGGREGATION_MODE in {"max_mean_l13", "max_last_l13"}:
        token = layer_13[token_pos]
        return torch.cat([token, token], dim=0)

    layer_12 = _safe_layer(hidden_states, 12)
    if AGGREGATION_MODE == "max_l12_l13":
        return torch.cat([layer_12[token_pos], layer_13[token_pos]], dim=0)

    if AGGREGATION_MODE == "max_mean_l12_l13":
        token_12 = layer_12[token_pos]
        token_13 = layer_13[token_pos]
        return torch.cat([token_12, token_12, token_13, token_13], dim=0)

    raise ValueError(f"Unknown AGGREGATION_MODE: {AGGREGATION_MODE}")


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into one fixed-size feature vector."""
    attention_mask = attention_mask.to(hidden_states.device)

    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    prompt_len = _next_prompt_length()

    if real_positions.numel() == 0:
        feature = _fallback_feature(hidden_states, -1)
        return torch.nan_to_num(feature).to(torch.float32)

    n_real = int(real_positions.numel())
    if prompt_len is None or prompt_len >= n_real:
        feature = _fallback_feature(hidden_states, real_positions[-1])
        return torch.nan_to_num(feature).to(torch.float32)

    response_positions = real_positions[prompt_len:]
    if response_positions.numel() == 0:
        feature = _fallback_feature(hidden_states, real_positions[-1])
        return torch.nan_to_num(feature).to(torch.float32)

    layer_13 = _safe_layer(hidden_states, 13)

    if AGGREGATION_MODE == "max_mean_l13":
        feature = torch.cat(
            [
                _max_pool(layer_13, response_positions),
                _mean_pool(layer_13, response_positions),
            ],
            dim=0,
        )
    elif AGGREGATION_MODE == "max_last_l13":
        feature = torch.cat(
            [
                _max_pool(layer_13, response_positions),
                layer_13[response_positions[-1]],
            ],
            dim=0,
        )
    elif AGGREGATION_MODE == "max_l12_l13":
        layer_12 = _safe_layer(hidden_states, 12)
        feature = torch.cat(
            [
                _max_pool(layer_12, response_positions),
                _max_pool(layer_13, response_positions),
            ],
            dim=0,
        )
    elif AGGREGATION_MODE == "max_mean_l12_l13":
        layer_12 = _safe_layer(hidden_states, 12)
        feature = torch.cat(
            [
                _max_pool(layer_12, response_positions),
                _mean_pool(layer_12, response_positions),
                _max_pool(layer_13, response_positions),
                _mean_pool(layer_13, response_positions),
            ],
            dim=0,
        )
    else:
        raise ValueError(f"Unknown AGGREGATION_MODE: {AGGREGATION_MODE}")

    return torch.nan_to_num(feature).to(torch.float32)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Optional geometric features are disabled by solution.py."""
    return torch.zeros(0, dtype=torch.float32, device=hidden_states.device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features."""
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
