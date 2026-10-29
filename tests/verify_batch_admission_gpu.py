"""Opt-in Qwen3 GPU regression; each revision/mode runs in a fresh process.

Example (with project dependencies installed):
    python tests/verify_batch_admission_gpu.py --model /path/to/Qwen3-0.6B \
        --base-root /path/to/unmodified/nano-vllm --output /tmp/batch-gpu.json
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def worker(args):
    sys.path.insert(0, str(Path(args.root).resolve()))
    import torch
    import flash_attn
    import transformers
    from nanovllm import LLM, SamplingParams
    import nanovllm.engine.llm_engine as engine_module

    assert torch.cuda.is_available(), "CUDA is required"
    assert Path(engine_module.__file__).resolve().is_relative_to(Path(args.root).resolve())
    llm = LLM(
        args.model, enforce_eager=args.mode == "eager", tensor_parallel_size=1,
        max_num_seqs=16, max_num_batched_tokens=512, max_model_len=256,
        gpu_memory_utilization=0.6,
    )
    params = SamplingParams(max_tokens=5, ignore_eos=True)

    # Compare a normal batch across revisions before intentionally failing any calls.
    torch.manual_seed(20261030)
    control = llm.generate(
        ["The capital of France is", "Count the first three positive integers:"],
        [SamplingParams(max_tokens=2, ignore_eos=True), params], use_tqdm=False,
    )
    assert [len(x["token_ids"]) for x in control] == [2, 5]
    digest = hashlib.sha256(json.dumps(control, sort_keys=True).encode()).hexdigest()

    def state():
        scheduler = llm.scheduler
        manager = scheduler.block_manager
        return (
            tuple(s.seq_id for s in scheduler.waiting),
            tuple(s.seq_id for s in scheduler.running),
            tuple((s.seq_id, s.num_tokens, s.num_cached_tokens, tuple(s.block_table))
                  for s in (*scheduler.waiting, *scheduler.running)),
            tuple(manager.free_block_ids), frozenset(manager.used_block_ids),
            tuple(b.ref_count for b in manager.blocks),
        )

    cases = []
    for existing in (False, True):
        for invalid_index in (0, 1, 2):
            assert llm.is_finished()
            if existing:
                llm.add_request("The opposite of hot is", SamplingParams(max_tokens=8, ignore_eos=True))
                llm.step()  # Real prefill leaves a running request and live KV blocks.
                llm.add_request("Two plus two equals", SamplingParams(max_tokens=3, ignore_eos=True))
            before = state()
            rng = torch.cuda.get_rng_state().clone()
            prompts = [llm.tokenizer.encode("One plus one equals")] * 3
            prompts[invalid_index] = []
            try:
                llm.generate(prompts, params, use_tqdm=False)
            except IndexError:
                pass
            else:
                raise AssertionError("expected the original empty-input error")
            unchanged = state() == before
            assert torch.equal(torch.cuda.get_rng_state(), rng), "failed admission ran inference"
            recovered = llm.generate(["The largest planet is"], params, use_tqdm=False)
            expected_lengths = [8, 3, 5] if existing else [5]
            isolated = [len(x["token_ids"]) for x in recovered] == expected_lengths
            assert llm.is_finished()
            assert not llm.scheduler.block_manager.used_block_ids
            cases.append({"existing_work": existing, "invalid_index": invalid_index,
                          "state_unchanged": unchanged, "next_call_isolated": isolated})

    assert llm.generate([], [], use_tqdm=False) == []
    return {
        "mode": args.mode, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "flash_attn": flash_attn.__version__, "transformers": transformers.__version__,
        "control_sha256": digest, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-root")
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("eager", "graph"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print("GPU_RESULT=" + json.dumps(worker(args)), flush=True)
        return
    candidate = Path(__file__).resolve().parents[1]
    roots = [("candidate", candidate)]
    if args.base_root:
        roots.insert(0, ("base", Path(args.base_root).resolve()))
    report = {}
    for label, root in roots:
        for mode in ("eager", "graph"):
            print(f"Running {label}/{mode} on GPU", flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--root", str(root), "--model", args.model, "--mode", mode]
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            rows = [line for line in result.stdout.splitlines() if line.startswith("GPU_RESULT=")]
            if len(rows) != 1:
                raise RuntimeError("worker did not return one GPU result: " + result.stdout + result.stderr)
            record = json.loads(rows[0].split("=", 1)[1])
            report[f"{label}/{mode}"] = record
            print(f"Finished {label}/{mode}", flush=True)
    for mode in ("eager", "graph"):
        record = report[f"candidate/{mode}"]
        assert all(c["state_unchanged"] and c["next_call_isolated"] for c in record["cases"])
        if args.base_root:
            base = report[f"base/{mode}"]
            assert base["control_sha256"] == record["control_sha256"], "normal output changed"
            for case in base["cases"]:
                expected = case["invalid_index"] == 0
                assert case["state_unchanged"] == expected
                assert case["next_call_isolated"] == expected
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: real GPU admission/recovery, existing work, normal output parity, and empty batches")


if __name__ == "__main__":
    main()
