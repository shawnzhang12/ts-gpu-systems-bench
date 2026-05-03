from __future__ import annotations

from src.train.runner import _batch_candidates


def test_batch_candidates_no_backoff() -> None:
    out = _batch_candidates(
        requested_batch_size=32,
        min_batch_size=4,
        auto_batch_backoff=False,
        max_batch_retries=3,
    )
    assert out == [32]


def test_batch_candidates_with_backoff() -> None:
    out = _batch_candidates(
        requested_batch_size=32,
        min_batch_size=4,
        auto_batch_backoff=True,
        max_batch_retries=3,
    )
    assert out == [32, 16, 8, 4]

