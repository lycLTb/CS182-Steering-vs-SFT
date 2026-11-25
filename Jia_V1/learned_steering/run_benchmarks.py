#!/usr/bin/env python3
"""Resumable Qwen3.5 text-only local evaluation; no network or judge API calls.

Uses published MMLU five-shot completion prompts and next-token A/B/C/D scoring.
Other benchmarks save raw completions for their appropriate external judges.
"""
import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import threading
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT.parent / "hardware-cache/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"


def read_jsonl(path):
    if not path.exists():
        return []
    text = path.read_text()
    # A monitor may read while another process is appending a record.
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def append(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data) + "\n")
        f.flush()


class Monitor:
    """RSS peak and MPS sampled maxima, explicitly not allocator peaks."""
    def __init__(self, device):
        import psutil
        self.device = device
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.peak_rss = 0
        self.peak_mps_sample = 0
        self.peak_driver_sample = 0
        self.minimum_available = psutil.virtual_memory().available

    def sample(self):
        import psutil
        import torch
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.minimum_available = min(self.minimum_available, psutil.virtual_memory().available)
        if self.device == "mps":
            self.peak_mps_sample = max(self.peak_mps_sample, torch.mps.current_allocated_memory())
            self.peak_driver_sample = max(self.peak_driver_sample, torch.mps.driver_allocated_memory())

    def run(self):
        while not self.stop.wait(.05):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sample()

    def result(self):
        return {"peak_rss_gib": self.peak_rss / 2**30,
                "mps_max_sampled_allocated_gib": self.peak_mps_sample / 2**30,
                "mps_max_sampled_driver_gib": self.peak_driver_sample / 2**30,
                "minimum_available_ram_gib": self.minimum_available / 2**30,
                "memory_sampling_interval_seconds": .05,
                "memory_note": "MPS sampled maximum, not CUDA peak; RSS and GPU share physical memory and must not be added"}


class Engine:
    def __init__(self, args):
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        self.torch = torch
        torch.manual_seed(args.seed)
        torch.set_num_threads(4)
        self.args = args
        self.device = args.device
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS inaccessible; run this script with GPU access outside the sandbox")
        self.tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, args.dtype)
        self.model = Qwen3_5ForCausalLM.from_pretrained(SNAPSHOT, local_files_only=True,
            dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
        self.model.requires_grad_(False)
        self.hook = None
        self.vector = None
        if args.variant != "baseline":
            checkpoint = ROOT / "checkpoints" / args.variant
            metadata = json.loads((checkpoint / "metadata.json").read_text())
            if metadata["method"] == "lora":
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, checkpoint).eval()
            else:
                vector = torch.load(checkpoint / "vector.pt", weights_only=True, map_location=args.device)
                self.install_vector(vector, metadata["layer"], metadata.get("scale", 1.0))
        self.sync()

    def sync(self):
        if self.device == "mps":
            self.torch.mps.synchronize()

    def install_vector(self, vector, layer, scale=1.0):
        if self.hook:
            self.hook.remove()
        self.vector = vector
        def steer(module, inputs, output):
            return output + scale * self.vector.to(output.dtype)
        self.hook = self.model.model.layers[layer].register_forward_hook(steer)

    def chat(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    def generate(self, messages, maximum):
        return self.generate_many([messages], maximum)[0]

    def generate_many(self, conversations, maximum):
        torch = self.torch
        prompts = [self.chat(messages) for messages in conversations]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        self.sync()
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=maximum,
                                      use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
        self.sync()
        elapsed = time.perf_counter() - start
        results = []
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for i, generated in enumerate(out[:, inputs.input_ids.shape[1]:].tolist()):
            stop = next((j + 1 for j, token in enumerate(generated) if token in eos), len(generated))
            ids = generated[:stop]
            results.append({"answer": self.tokenizer.decode(ids, skip_special_tokens=True),
                "input_tokens": int(inputs.attention_mask[i].sum().item()), "output_tokens": len(ids),
                "batch_wall_seconds": elapsed, "batch_size": len(conversations),
                "hit_output_limit": len(ids) == maximum and (not ids or ids[-1] not in eos)})
        return results


def metadata(args):
    import torch
    import psutil
    return {"model": "Qwen/Qwen3.5-0.8B", "revision": SNAPSHOT.name,
            "variant": args.variant, "device": args.device, "dtype": args.dtype,
            "seed": args.seed, "platform": platform.platform(), "machine": platform.machine(),
            "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "host_ram_gib": psutil.virtual_memory().total / 2**30,
            "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate")},
            "attention": "sdpa", "thinking": False, "generation": "greedy",
            "max_new_tokens": args.max_new_tokens, "mmlu_shots": 5,
            "maximum_batch_size": getattr(args, "batch_size", 4),
            "data_manifest_sha256": hashlib.sha256((ROOT / "data/manifest.json").read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def probe(engine, args):
    torch = engine.torch
    result = {"metadata": metadata(args), "parameter_count": engine.model.num_parameters(), "inference": []}
    ids = engine.tokenizer("Attention combines information from earlier tokens. " * 512,
                           return_tensors="pt").input_ids.to(engine.device)
    for length in (128, 512, 2048):
        x = ids[:, :length]
        mask = torch.ones_like(x)
        gen = dict(input_ids=x, attention_mask=mask, do_sample=False, use_cache=True,
                   pad_token_id=engine.tokenizer.pad_token_id)
        with Monitor(engine.device) as mem, torch.inference_mode():
            engine.model.generate(**gen, min_new_tokens=4, max_new_tokens=4)
            engine.sync()
            times = []
            for _ in range(3):
                engine.sync()
                start = time.perf_counter()
                out = engine.model.generate(**gen, min_new_tokens=64, max_new_tokens=64)
                engine.sync()
                assert out.shape[1] == length + 64
                times.append(time.perf_counter() - start)
        row = {"input_tokens": length, "output_tokens": 64, "batch_size": 1,
               "repeats": 3, "seconds": times, "median_seconds": statistics.median(times),
               "median_tokens_per_second": 64 / statistics.median(times), **mem.result()}
        result["inference"].append(row)
        save_json(ROOT / "results" / args.variant / "compute.json", result)
        print(json.dumps({"event": "inference_probe", **row}), flush=True)
    return result


def mmlu_prompt(row, dev, shots=5):
    def example(r, answered):
        s = r["question"] + "".join(f"\n{'ABCD'[i]}. {c}" for i, c in enumerate(r["choices"])) + "\nAnswer:"
        return s + (" " + "ABCD"[r["answer"]] + "\n\n" if answered else "")
    subject = row["subject"]
    prompt = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"
    return prompt + "".join(example(r, True) for r in dev[subject][:shots]) + example(row, False)


def evaluate(engine, args):
    import csv
    torch = engine.torch
    output = ROOT / "results" / args.variant
    output.mkdir(parents=True, exist_ok=True)
    meta = metadata(args)
    with Monitor(engine.device) as mem:
        for bench in args.benchmarks:
            meta_path = output / f"{bench}_metadata.json"
            if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
                raise RuntimeError("Result metadata changed; use a new variant/output folder to avoid mixing runs")
            save_json(meta_path, meta)
            path = output / f"{bench}.jsonl"
            if path.exists():
                raw = path.read_bytes()
                if raw and not raw.endswith(b"\n"):
                    boundary = raw.rfind(b"\n") + 1
                    path.with_suffix(".interrupted-tail").write_bytes(raw[boundary:])
                    path.write_bytes(raw[:boundary])
            previous = read_jsonl(path)
            done = {r["id"] for r in previous}
            rows = []
            if bench == "mmlu":
                dev = {}
                for r in read_jsonl(ROOT / "data/mmlu_dev.jsonl"):
                    dev.setdefault(r["subject"], []).append(r)
                rows = read_jsonl(ROOT / "data/mmlu_test.jsonl")
                answer_ids = [engine.tokenizer.encode(" " + c, add_special_tokens=False) for c in "ABCD"]
                if any(len(x) != 1 for x in answer_ids):
                    raise ValueError("A/B/C/D must each tokenize to one token for this MMLU implementation")
                answer_ids = [x[0] for x in answer_ids]
            elif bench in ("xstest", "harmbench"):
                with (ROOT / f"data/{bench}.csv").open() as f:
                    rows = list(csv.DictReader(f))
            elif bench == "mt_bench":
                rows = read_jsonl(ROOT / "data/mt_bench.jsonl")
            elif bench == "wildjailbreak":
                with (ROOT / "data/wildjailbreak_eval.tsv").open() as f:
                    rows = list(csv.DictReader(f, delimiter="\t"))
            else:
                raise ValueError(bench)
            pending = []
            for i, row in enumerate(rows):
                key = str(i) if bench not in ("mt_bench", "harmbench") else str(row.get("question_id", row.get("BehaviorID")))
                if key not in done:
                    pending.append((key, row))
            if args.limit:
                pending = pending[:max(0, args.limit - len(done))]
            # Shorter prompts together reduce padding; all IDs remain source IDs.
            if bench == "mmlu":
                pending.sort(key=lambda kr: len(mmlu_prompt(kr[1], dev)))
            index = 0
            while index < len(pending):
                candidates = pending[index:index + args.batch_size]
                # Bound padded input tokens. Never truncate benchmark prompts.
                def prompt_for(row):
                    if bench == "mmlu":
                        return mmlu_prompt(row, dev)
                    if bench == "mt_bench":
                        return engine.chat([{"role": "user", "content": row["turns"][0]}])
                    p = row["prompt"] if bench == "xstest" else row["Behavior"] if bench == "harmbench" else row.get("adversarial") or row.get("vanilla")
                    if bench == "harmbench" and row.get("ContextString"):
                        p = row["ContextString"] + "\n\n---\n\n" + p
                    return engine.chat([{"role": "user", "content": p}])
                lengths = [len(engine.tokenizer.encode(prompt_for(r), add_special_tokens=False)) for _, r in candidates]
                maximum = max(lengths)
                count = max(1, min(len(candidates), 4096 // max(1, maximum)))
                batch = candidates[:count]
                entries = []
                if bench == "mmlu":
                    prompts = [mmlu_prompt(r, dev) for _, r in batch]
                    inputs = engine.tokenizer(prompts, return_tensors="pt", padding=True).to(engine.device)
                    inputs["position_ids"] = (inputs.attention_mask.long().cumsum(-1) - 1).masked_fill(inputs.attention_mask == 0, 0)
                    engine.sync()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        scores = engine.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, answer_ids].float()
                        probabilities = scores.softmax(-1).cpu().tolist()
                    engine.sync()
                    elapsed = time.perf_counter() - start
                    for j, ((key, row), probs) in enumerate(zip(batch, probabilities)):
                        prediction = max(range(4), key=probs.__getitem__)
                        entries.append({"id": key, "subject": row["subject"], "prediction": prediction, "gold": row["answer"],
                             "correct": prediction == row["answer"], "choice_probabilities": probs,
                             "input_tokens": int(inputs.attention_mask[j].sum().item()), "batch_wall_seconds": elapsed,
                             "batch_size": len(batch)})
                elif bench == "mt_bench":
                    conversations = [[] for _ in batch]
                    turns = [[] for _ in batch]
                    for t in range(2):
                        for j, (_, row) in enumerate(batch):
                            conversations[j].append({"role": "user", "content": row["turns"][t]})
                        answers = engine.generate_many(conversations, args.max_new_tokens)
                        for j, answer in enumerate(answers):
                            turns[j].append(answer)
                            conversations[j].append({"role": "assistant", "content": answer["answer"]})
                    entries = [{"id": key, "question_id": row["question_id"], "category": row["category"],
                                "turns": turns[j], "judge_status": "pending_external_judge"}
                               for j, (key, row) in enumerate(batch)]
                else:
                    conversations = []
                    for _, row in batch:
                        p = row["prompt"] if bench == "xstest" else row["Behavior"] if bench == "harmbench" else row.get("adversarial") or row.get("vanilla")
                        if bench == "harmbench" and row.get("ContextString"):
                            p = row["ContextString"] + "\n\n---\n\n" + p
                        conversations.append([{"role": "user", "content": p}])
                    answers = engine.generate_many(conversations, args.max_new_tokens)
                    for (key, row), answer in zip(batch, answers):
                        entry = {"id": key, **answer, "judge_status": "pending_external_judge"}
                        if bench == "xstest":
                            entry.update(source_id=row["id"], type=row["type"], label=row["label"])
                        elif bench == "harmbench":
                            entry.update(behavior_id=row["BehaviorID"], functional_category=row["FunctionalCategory"])
                        else:
                            entry["data_type"] = row.get("data_type")
                        entries.append(entry)
                for entry in entries:
                    append(path, entry)
                    done.add(entry["id"])
                index += len(batch)
                if len(done) % 8 < len(batch) or len(done) == len(rows):
                    status = {"event": "progress", "variant": args.variant, "benchmark": bench, "done": len(done),
                              "total": len(rows), "updated_unix": time.time(), **mem.result()}
                    save_json(output / "progress.json", status)
                    print(json.dumps(status), flush=True)
                if len(done) % 100 == 0 and engine.device == "mps":
                    torch.mps.empty_cache()
            completed = read_jsonl(path)
            summary = {"benchmark": bench, "completed": len(completed), "total": len(rows),
                       "status": "complete" if len(completed) == len(rows) else "partial", **mem.result()}
            if bench == "mmlu" and completed:
                summary["accuracy"] = sum(r["correct"] for r in completed) / len(completed)
                subjects = sorted({r["subject"] for r in completed})
                summary["subject_accuracy"] = {s: statistics.mean(r["correct"] for r in completed if r["subject"] == s) for s in subjects}
            else:
                summary["score_status"] = "pending_external_judge"
            save_json(output / f"{bench}_summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["probe", "evaluate"])
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], help="default FP32 for MMLU, BF16 otherwise")
    parser.add_argument("--seed", type=int, default=182)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu", "xstest", "harmbench", "mt_bench", "wildjailbreak"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="0 means full dataset; nonzero is explicitly partial")
    args = parser.parse_args()
    args.dtype = args.dtype or ("float32" if args.mode == "evaluate" and "mmlu" in args.benchmarks else "bfloat16")
    engine = Engine(args)
    if args.mode == "probe":
        probe(engine, args)
    else:
        evaluate(engine, args)


if __name__ == "__main__":
    main()
