"""Unit tests for the tinker_cp patches, grounded in Miles' real CP offset math."""

from __future__ import annotations

import importlib.machinery
import sys
from argparse import Namespace
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


class _StubModule(type(sys)):
    """Module placeholder whose attribute access yields mocks."""

    def __getattr__(self, name):
        return MagicMock(name=f"{self.__name__}.{name}")


def _stub_missing(name: str) -> None:
    module = _StubModule(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    module.__path__ = []
    sys.modules[name] = module


# tinker_cp imports Miles' fsdp actor chain, which drags in miles' object
# store, ray and sglang (none installed in the CPU test env). Stub the missing
# pieces while importing, then remove them so sibling tests see the real env.
_stubbed: list[str] = []
try:
    for _ in range(300):
        try:
            from miles.backends.training_utils.cp_utils import (
                get_logits_and_tokens_offset_with_cp as _real_offsets,
            )
            from spindle.backends.miles_runtime import tinker_cp

            break
        except ModuleNotFoundError as exc:
            if exc.name is None:
                raise
            _stub_missing(exc.name)
            _stubbed.append(exc.name)
            sys.modules.pop("spindle.backends.miles_runtime.tinker_cp", None)
    else:
        raise RuntimeError("tinker_cp import chain did not converge")
finally:
    # Never leak stubs into sibling tests, even on a failed import.
    for _name in _stubbed:
        sys.modules.pop(_name, None)

TOTAL = 16
RESPONSE = 12
PROMPT = TOTAL - RESPONSE
CP_SIZE = 2


def _offsets(rank):
    return _real_offsets(TOTAL, RESPONSE, "thd", None, cp_rank=rank, cp_size=CP_SIZE)


def _shard(rank: int, *, requires_grad: bool = False) -> torch.Tensor:
    _, _, logits_offset, _ = _offsets(rank)
    length = (logits_offset[0][1] - logits_offset[0][0]) + (
        logits_offset[1][1] - logits_offset[1][0]
    )
    start = 100.0 * (rank + 1)
    return torch.arange(
        start, start + length, dtype=torch.float32, requires_grad=requires_grad
    )


def _expected_pad(shard: torch.Tensor, rank: int) -> torch.Tensor:
    """Independent construction: shard values at the rank's real zigzag windows."""
    _, _, logits_offset, _ = _offsets(rank)
    padded = torch.zeros(RESPONSE)
    start_0 = logits_offset[0][0] - (PROMPT - 1)
    end_0 = logits_offset[0][1] - (PROMPT - 1)
    start_1 = logits_offset[1][0] - (PROMPT - 1)
    end_1 = logits_offset[1][1] - (PROMPT - 1)
    padded[start_0:end_0] = shard[: end_0 - start_0]
    padded[start_1:end_1] = shard[end_0 - start_0 :]
    return padded


def _make_fakes(monkeypatch, rank: int):
    """Install tinker_cp against fake upstream modules bound to one CP rank."""
    received = {}

    def get_log_probs_and_entropy(logits, **kwargs):
        received["logits"] = logits
        received["kwargs"] = kwargs
        return received["out"]

    def slice_log_prob_with_cp(
        log_prob, total_length, response_length, qkv_format="thd", max_token_len=None
    ):
        raise AssertionError("the patch must replace this function")

    cp_utils = SimpleNamespace(
        get_logits_and_tokens_offset_with_cp=partial(
            _real_offsets, cp_rank=rank, cp_size=CP_SIZE
        ),
        slice_log_prob_with_cp=slice_log_prob_with_cp,
    )
    logit_processors = SimpleNamespace(
        get_log_probs_and_entropy=get_log_probs_and_entropy
    )
    dependents = {
        name: SimpleNamespace(slice_log_prob_with_cp=slice_log_prob_with_cp)
        for name in ("data", "mm_data", "math_utils")
    }
    bound_at_import = {
        name: SimpleNamespace(get_log_probs_and_entropy=get_log_probs_and_entropy)
        for name in ("loss", "tinker_losses", "megatron_actor", "fsdp_actor")
    }
    state = SimpleNamespace(cp=SimpleNamespace(size=CP_SIZE, rank=rank, group=None))
    monkeypatch.setattr(tinker_cp, "cp_utils", cp_utils)
    monkeypatch.setattr(tinker_cp, "logit_processors", logit_processors)
    monkeypatch.setattr(tinker_cp, "get_parallel_state", lambda: state)
    for name, module in {**dependents, **bound_at_import}.items():
        monkeypatch.setattr(tinker_cp, name, module)
    return received, cp_utils, logit_processors, dependents, bound_at_import


def _call(patched, shard, *, allgather_cp=False):
    return patched(
        torch.zeros(4, 4),
        args=Namespace(allgather_cp=allgather_cp, qkv_format="thd"),
        unconcat_tokens=[torch.zeros(4)],
        total_lengths=[TOTAL],
        response_lengths=[RESPONSE],
        with_entropy=True,
        entropy_requires_grad=False,
        non_loss_data=True,
        max_seq_lens=None,
        rollout_sampling_mask=None,
    )


def test_passthrough_forwards_every_argument_when_cp_size_one(monkeypatch):
    received, cp_utils, logit_processors, _, _ = _make_fakes(monkeypatch, rank=0)
    monkeypatch.setattr(
        tinker_cp,
        "get_parallel_state",
        lambda: SimpleNamespace(cp=SimpleNamespace(size=1, rank=0, group=None)),
    )
    tinker_cp.install()
    sentinel = {"log_probs": ["untouched"]}
    received["out"] = sentinel

    logits = torch.zeros(4, 4)
    result = logit_processors.get_log_probs_and_entropy(
        logits,
        args=Namespace(allgather_cp=False, qkv_format="thd"),
        unconcat_tokens=[torch.zeros(4)],
        total_lengths=[TOTAL],
        response_lengths=[RESPONSE],
        with_entropy=True,
        entropy_requires_grad=False,
        non_loss_data=False,
        max_seq_lens=[TOTAL],
        rollout_sampling_mask=None,
    )

    assert result is sentinel
    assert received["logits"] is logits
    assert received["kwargs"] == {
        "args": Namespace(allgather_cp=False, qkv_format="thd"),
        "unconcat_tokens": received["kwargs"]["unconcat_tokens"],
        "total_lengths": [TOTAL],
        "response_lengths": [RESPONSE],
        "with_entropy": True,
        "entropy_requires_grad": False,
        "non_loss_data": False,
        "max_seq_lens": [TOTAL],
        "rollout_sampling_mask": None,
    }


def test_passthrough_when_allgather_cp(monkeypatch):
    received, _, logit_processors, _, _ = _make_fakes(monkeypatch, rank=0)
    tinker_cp.install()
    sentinel = {"log_probs": ["untouched"]}
    received["out"] = sentinel

    result = _call(
        logit_processors.get_log_probs_and_entropy, _shard(0), allgather_cp=True
    )

    assert result is sentinel


def test_slice_replacement_is_identity_and_repoints_dependents(monkeypatch):
    _, cp_utils, _, dependents, _ = _make_fakes(monkeypatch, rank=0)
    tinker_cp.install()

    replacement = cp_utils.slice_log_prob_with_cp
    assert getattr(replacement, "__spindle_unslices_cp__", False)
    for module in dependents.values():
        assert module.slice_log_prob_with_cp is replacement
    tensor = torch.zeros(5)
    assert replacement(tensor, TOTAL, RESPONSE) is tensor


def test_gather_matches_real_upstream_zigzag_geometry(monkeypatch):
    """Each rank's reassembled log_probs equal the sum of both ranks' pads, and
    slicing that sum with Miles' real offsets reproduces the original shard."""
    pads = {rank: _expected_pad(_shard(rank), rank) for rank in range(CP_SIZE)}
    results = {}
    for rank in range(CP_SIZE):
        received, _, logit_processors, _, _ = _make_fakes(monkeypatch, rank=rank)
        tinker_cp.install()
        shard = _shard(rank, requires_grad=True)
        other_pad = pads[1 - rank]
        monkeypatch.setattr(
            tinker_cp.dist,
            "all_reduce",
            lambda t, group=None, _pad=other_pad: t.add_(_pad),
        )
        received["out"] = {"log_probs": [shard]}
        results[rank] = _call(logit_processors.get_log_probs_and_entropy, shard)[
            "log_probs"
        ][0]
        monkeypatch.undo()

    summed = pads[0] + pads[1]
    assert summed.shape == (RESPONSE,)
    for rank in range(CP_SIZE):
        assert torch.allclose(results[rank], summed)

    # Round-trip: Miles' real slicing of the gathered rows returns each shard.
    for rank in range(CP_SIZE):
        _, _, logits_offset, _ = _offsets(rank)
        chunk_0 = summed[
            logits_offset[0][0] - (PROMPT - 1) : logits_offset[0][1] - (PROMPT - 1)
        ]
        chunk_1 = summed[
            logits_offset[1][0] - (PROMPT - 1) : logits_offset[1][1] - (PROMPT - 1)
        ]
        recovered = torch.cat([chunk_0, chunk_1], dim=0)
        assert torch.equal(recovered, _shard(rank))


def test_pad_local_shard_places_only_the_ranks_windows(monkeypatch):
    rank = 0
    _, cp_utils, _, _, _ = _make_fakes(monkeypatch, rank=rank)
    monkeypatch.setattr(tinker_cp, "cp_utils", cp_utils)
    shard = _shard(rank)

    padded = tinker_cp._pad_local_shard(
        shard, TOTAL, RESPONSE, qkv_format="thd", max_seq_len=None
    )

    assert padded.shape == (RESPONSE,)
    assert torch.equal(padded.detach(), _expected_pad(shard, rank))


def test_install_is_idempotent(monkeypatch):
    _, cp_utils, logit_processors, _, _ = _make_fakes(monkeypatch, rank=0)
    tinker_cp.install()
    gathered = logit_processors.get_log_probs_and_entropy
    sliced = cp_utils.slice_log_prob_with_cp

    tinker_cp.install()

    assert logit_processors.get_log_probs_and_entropy is gathered
    assert cp_utils.slice_log_prob_with_cp is sliced


def test_pad_local_shard_fails_loudly_on_an_inverted_window(monkeypatch):
    """Demonstrated pinned-upstream behavior: when the prompt leaves a CP rank
    no logits in its first zigzag window, the real offset function returns
    that window inverted and the real slice yields a malformed shard. Whether
    Miles supports such geometry is not established here; _pad_local_shard
    must at least refuse the malformed shard loudly, not misplace it."""
    rank = 0
    _, cp_utils, _, _, _ = _make_fakes(monkeypatch, rank=rank)
    monkeypatch.setattr(tinker_cp, "cp_utils", cp_utils)

    # TOTAL=16 / RESPONSE=10: rank 0's first window is (5, 4) under the real
    # upstream offsets, and Miles' slice produces a 12-row "shard" for a
    # 10-token response.
    malformed = torch.arange(12.0)
    with pytest.raises(AssertionError):
        tinker_cp._pad_local_shard(
            malformed, 16, 10, qkv_format="thd", max_seq_len=None
        )
