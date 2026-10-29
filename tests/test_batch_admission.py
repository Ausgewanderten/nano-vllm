"""Batch preparation regressions using real scheduling and synthetic inference.

Requires nano-vllm dependencies and pytest, but no GPU or model weights.
"""

from types import SimpleNamespace

import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class Tokenizer:
    def encode(self, text):
        return [int(token) for token in text.split()]

    def decode(self, tokens):
        return " ".join(map(str, tokens))


class SyntheticRunner:
    def __init__(self):
        self.calls = 0

    def call(self, method, seqs, is_prefill):
        assert method == "run"
        self.calls += 1
        assert self.calls <= 20, "scheduler did not make progress"
        return [seq.last_token + 1 for seq in seqs]


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 256)
    config = SimpleNamespace(
        max_num_seqs=8, max_num_batched_tokens=1024, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=4,
    )
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = Scheduler(config)
    engine.tokenizer = Tokenizer()
    engine.model_runner = SyntheticRunner()
    return engine


@pytest.mark.parametrize("invalid_index", [0, 1, 2])
@pytest.mark.parametrize("existing_work", [False, True])
def test_failed_batch_preserves_scheduler_and_next_call(engine, invalid_index, existing_work):
    if existing_work:
        engine.add_request([10], SamplingParams(max_tokens=3, ignore_eos=True))
        engine.step()
        engine.add_request([20], SamplingParams(max_tokens=2, ignore_eos=True))
    waiting = list(engine.scheduler.waiting)
    running = list(engine.scheduler.running)
    manager = engine.scheduler.block_manager
    free_blocks = list(manager.free_block_ids)
    refs = [block.ref_count for block in manager.blocks]
    model_calls = engine.model_runner.calls
    prompts = [[100], [200], [300]]
    prompts[invalid_index] = []

    with pytest.raises(IndexError):
        engine.generate(prompts, SamplingParams(max_tokens=1), use_tqdm=False)

    assert list(engine.scheduler.waiting) == waiting
    assert list(engine.scheduler.running) == running
    assert list(manager.free_block_ids) == free_blocks
    assert [block.ref_count for block in manager.blocks] == refs
    assert engine.model_runner.calls == model_calls
    results = engine.generate([[400]], SamplingParams(max_tokens=1), use_tqdm=False)
    expected = [[11, 12, 13], [21, 22], [401]] if existing_work else [[401]]
    assert [result["token_ids"] for result in results] == expected


def test_tokenization_failure_does_not_admit_earlier_requests(engine):
    with pytest.raises(ValueError):
        engine.generate(["100", "bad"], SamplingParams(max_tokens=1), use_tqdm=False)
    assert engine.is_finished()
    assert engine.model_runner.calls == 0
    results = engine.generate(["200"], SamplingParams(max_tokens=2), use_tqdm=False)
    assert [result["token_ids"] for result in results] == [[201, 202]]


def test_sequence_construction_failure_does_not_admit_earlier_requests(engine):
    with pytest.raises(AttributeError):
        engine.generate([[100], [200]], [SamplingParams(max_tokens=1), object()], use_tqdm=False)
    assert engine.is_finished()
    assert engine.model_runner.calls == 0


def test_valid_batch_keeps_per_request_parameters_and_order(engine):
    results = engine.generate(
        ["100", "200"], [SamplingParams(max_tokens=1), SamplingParams(max_tokens=2)], use_tqdm=False,
    )
    assert results == [
        {"text": "101", "token_ids": [101]},
        {"text": "201 202", "token_ids": [201, 202]},
    ]


def test_manual_add_request_still_queues_and_runs(engine):
    engine.add_request([100], SamplingParams(max_tokens=1))
    seq = engine.scheduler.waiting[0]
    outputs, _ = engine.step()
    assert outputs == [(seq.seq_id, [101])]
    assert engine.is_finished()


def test_empty_batch_remains_valid(engine):
    assert engine.generate([], [], use_tqdm=False) == []
    assert engine.model_runner.calls == 0
