"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
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
PROBE_LAYER = 13

_PROMPT_LENS: list[int] | None = None
_AGGREGATE_COUNTER = 0


def _load_prompt_lengths() -> list[int]:
    """Tokenize prompts once, matching solution.py train-then-test ordering."""
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
    return _PROMPT_LENS


def _next_prompt_length() -> int | None:
    global _AGGREGATE_COUNTER
    prompt_lens = _load_prompt_lengths()
    prompt_len = (
        prompt_lens[_AGGREGATE_COUNTER]
        if _AGGREGATE_COUNTER < len(prompt_lens)
        else None
    )
    _AGGREGATE_COUNTER += 1
    return prompt_len


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    attention_mask = attention_mask.to(hidden_states.device)

    n_layers = hidden_states.shape[0]
    layer_idx = min(PROBE_LAYER, n_layers - 1)
    layer = hidden_states[layer_idx]
    prompt_len = _next_prompt_length()

    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    if real_positions.numel() == 0:
        feature = layer[-1]
        return torch.nan_to_num(feature).to(torch.float32)

    n_real = int(real_positions.numel())
    if prompt_len is None or prompt_len >= n_real:
        feature = layer[real_positions[-1]]
        return torch.nan_to_num(feature).to(torch.float32)

    response_positions = real_positions[prompt_len:]
    if response_positions.numel() == 0:
        feature = layer[real_positions[-1]]
        return torch.nan_to_num(feature).to(torch.float32)

    response_states = layer.index_select(0, response_positions)
    feature = response_states.max(dim=0).values

    return torch.nan_to_num(feature).to(torch.float32)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0, dtype=torch.float32, device=hidden_states.device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
