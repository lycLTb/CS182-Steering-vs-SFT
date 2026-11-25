#!/usr/bin/env python3
"""Train proposal methods with balanced, disjoint WildJailbreak splits.

Primary checkpoints use equal cumulative training-time budgets (15/60/180 s).
Secondary snapshots at exactly 16/64/256 examples support data-matched analysis.
These controls are distinct: equal time generally implies unequal data exposure.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import time

from run_benchmarks import Engine, Monitor, ROOT, save_json, read_jsonl, metadata

SEED = 182
LEVELS = [("small", 16, 15), ("medium", 64, 60), ("large", 256, 180)]


def prepare_split():
    split_path = ROOT / "data/training_split.json"
    if split_path.exists():
        return json.loads(split_path.read_text())
    groups = defaultdict(list)
    # Group by underlying vanilla query before splitting to prevent adversarial
    # paraphrases of the same query crossing train/validation boundaries.
    rng = random.Random(SEED)
    with (ROOT / "data/wildjailbreak_train.tsv").open() as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
            r["source_row"] = i
            groups[r["vanilla"].strip().casefold()].append(r)
    keys = sorted(groups)
    rng.shuffle(keys)
    val_keys = set(keys[:max(100, len(keys) // 20)])
    pools = {split: defaultdict(list) for split in ("train", "validation")}
    for key, records in groups.items():
        split = "validation" if key in val_keys else "train"
        for r in records:
            pools[split][r["data_type"]].append(r)
    chosen = {}
    for split, per_type in pools.items():
        count = 64 if split == "train" else 8
        for records in per_type.values():
            rng.shuffle(records)
        types = sorted(per_type)
        assert len(types) == 4, types
        chosen[split] = [per_type[t][i] for i in range(count) for t in types]
    train_keys = {r["vanilla"].strip().casefold() for r in chosen["train"]}
    validation_keys = {r["vanilla"].strip().casefold() for r in chosen["validation"]}
    assert not train_keys & validation_keys
    chosen["metadata"] = {"seed": SEED, "grouping": "casefolded vanilla prompt", "train_examples": 256,
                          "validation_examples": 32, "balanced_data_types": types,
                          "source_sha256": hashlib.sha256((ROOT / "data/wildjailbreak_train.tsv").read_bytes()).hexdigest()}
    save_json(split_path, chosen)
    return chosen


def encode(engine, row):
    torch = engine.torch
    prompt = row["adversarial"] or row["vanilla"]
    text = engine.chat([{"role": "user", "content": prompt}])
    prefix = engine.tokenizer.encode(text, add_special_tokens=False)
    suffix = engine.tokenizer.encode(row["completion"] + engine.tokenizer.eos_token, add_special_tokens=False)
    # Preserve the assistant prefix and the end of long adversarial prompts.
    prefix, suffix = prefix[-256:], suffix[:128]
    return {"input_ids": torch.tensor([prefix + suffix], device=engine.device),
            "target": torch.tensor(suffix, device=engine.device),
            "prefix_length": len(prefix), "response_tokens": len(suffix)}


def response_loss(engine, example):
    # Only project supervised response positions into the large vocabulary.
    # This is exactly causal response-only CE, with no second label shift.
    torch = engine.torch
    out = engine.model.model(input_ids=example["input_ids"], use_cache=False).last_hidden_state
    start = example["prefix_length"] - 1
    hidden = out[0, start:start + example["response_tokens"]]
    logits = engine.model.lm_head(hidden)
    return torch.nn.functional.cross_entropy(logits.float(), example["target"])


def create_method(engine, method):
    torch = engine.torch
    if method == "learned":
        engine.vector = torch.nn.Parameter(torch.zeros(engine.model.config.hidden_size, device=engine.device, dtype=torch.float32))
        engine.install_vector(engine.vector, 11)
        params = [engine.vector]
    elif method == "lora":
        from peft import LoraConfig, get_peft_model
        targets = [name for name, module in engine.model.named_modules()
                   if isinstance(module, torch.nn.Linear) and ".self_attn." in name
                   and name.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")]
        assert len(targets) == 24, targets  # six full-attention blocks
        engine.model = get_peft_model(engine.model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
            bias="none", target_modules=targets, task_type="CAUSAL_LM"))
        params = [p for p in engine.model.parameters() if p.requires_grad]
    else:
        raise ValueError(method)
    engine.model.eval()  # dropout disabled for matched objectives
    engine.model.config.use_cache = False
    return params


def loss(engine, example):
    # PEFT forwards unknown attributes to its base model. Unwrap its .model once
    # so response_loss addresses the text decoder and tied language head.
    if hasattr(engine.model, "peft_config"):
        wrapper = engine.model
        engine.model = wrapper.get_base_model()
        try:
            return response_loss(engine, example)
        finally:
            engine.model = wrapper
    return response_loss(engine, example)


def validate(engine, examples):
    values = []
    with engine.torch.no_grad():
        for x in examples:
            values.append(float(loss(engine, x).item()))
    engine.sync()
    return sum(values) / len(values)


def save_checkpoint(engine, name, method, info):
    folder = ROOT / "checkpoints" / name
    folder.mkdir(parents=True, exist_ok=True)
    if method == "lora":
        engine.model.save_pretrained(folder)
    else:
        engine.torch.save(engine.vector.detach().cpu(), folder / "vector.pt")
    save_json(folder / "metadata.json", {"method": method, "layer": 11, "scale": 1.0,
        "training_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **info})


def train(engine, args, split):
    torch = engine.torch
    training = [encode(engine, r) for r in split["train"]]
    validation = [encode(engine, r) for r in split["validation"]]
    if args.method == "dom":
        sums = {"harmful": None, "benign": None}
        counts = {"harmful": 0, "benign": 0}
        captured = []
        def capture(module, inputs, output):
            captured.append(output[0, -1].detach().float())
        hook = engine.model.model.layers[11].register_forward_hook(capture)
        start = time.perf_counter()
        for level, exposure, budget in LEVELS:
            old = sum(counts.values())
            with Monitor(engine.device) as mem, torch.no_grad():
                for row in split["train"][old:exposure]:
                    prompt = row["adversarial"] or row["vanilla"]
                    ids = engine.tokenizer.encode(engine.chat([{"role": "user", "content": prompt}]), add_special_tokens=False)[-256:]
                    captured.clear()
                    engine.model.model(input_ids=torch.tensor([ids], device=engine.device), use_cache=False)
                    group = "harmful" if row["data_type"].endswith("_harmful") else "benign"
                    vec = captured[0]
                    sums[group] = vec.clone() if sums[group] is None else sums[group] + vec
                    counts[group] += 1
            engine.sync()
            extraction_time = time.perf_counter() - start
            direction = sums["harmful"] / counts["harmful"] - sums["benign"] / counts["benign"]
            # Validation labels are never used to compute a direction. Select
            # the coefficient on a disjoint training-validation partition.
            hook.remove()
            search_start = time.perf_counter()
            candidates = []
            for scale in (0.5, 1.0, 2.0):
                engine.install_vector(direction * scale, 11)
                candidates.append({"scale": scale, "validation_response_nll": validate(engine, validation[:8])})
            best = min(candidates, key=lambda x: x["validation_response_nll"])
            engine.vector = direction * best["scale"]
            search_seconds = time.perf_counter() - search_start
            if search_seconds > 60:
                raise RuntimeError("DoM coefficient search exceeded its common 60-second allowance")
            info = {"level": level, "data_exposure": exposure, "training_budget_seconds": budget,
                    "extraction_seconds": extraction_time, "hyperparameter_search_seconds": search_seconds,
                    "search": candidates, "hyperparameter_search_budget_seconds": 60,
                    "chosen_coefficient": best["scale"], "trainable_parameters": 0,
                    "stored_parameters": direction.numel(), "direction": "mean harmful prompt activation minus mean benign prompt activation at last prompt token",
                    "validation_response_nll": best["validation_response_nll"], **mem.result(), "model": metadata(args)}
            name = f"dom_{level}"
            save_checkpoint(engine, name, "dom", info)
            print(json.dumps({"event": "checkpoint", "variant": name, **info}), flush=True)
            engine.hook.remove()
            engine.hook = None
            hook = engine.model.model.layers[11].register_forward_hook(capture)
            # Exclude coefficient selection from cumulative extraction cost.
            start += search_seconds
        hook.remove()
        return
    params = create_method(engine, args.method)
    count = sum(p.numel() for p in params)
    # Three candidates, identical eight-example exposure, no benchmark-set
    # feedback. Record actual time separately from final training.
    rates = (0.01, 0.05, 0.1) if args.method == "learned" else (1e-4, 3e-4, 1e-3)
    initial = [p.detach().clone() for p in params]
    search_start = time.perf_counter()
    candidates = []
    for rate in rates:
        with torch.no_grad():
            for p, value in zip(params, initial):
                p.copy_(value)
        opt = torch.optim.AdamW(params, lr=rate, weight_decay=0, foreach=False)
        for x in training[:8]:
            opt.zero_grad(set_to_none=True)
            objective = loss(engine, x)
            if not torch.isfinite(objective).item():
                raise RuntimeError("Nonfinite search loss")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        candidates.append({"learning_rate": rate, "validation_response_nll": validate(engine, validation[:8])})
    best = min(candidates, key=lambda x: x["validation_response_nll"])
    engine.sync()
    search_seconds = time.perf_counter() - search_start
    if search_seconds > 60:
        raise RuntimeError("Hyperparameter search exceeded its common 60-second allowance")
    with torch.no_grad():
        for p, value in zip(params, initial):
            p.copy_(value)
    opt = torch.optim.AdamW(params, lr=best["learning_rate"], weight_decay=0, foreach=False)
    steps = 0
    elapsed = 0.0
    losses = []
    for level, exposure, budget in LEVELS:
        with Monitor(engine.device) as mem:
            while elapsed < budget:
                x = training[steps % len(training)]
                engine.sync()
                start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                objective = loss(engine, x)
                if not torch.isfinite(objective).item():
                    raise RuntimeError("Nonfinite training loss")
                objective.backward()
                if not any(p.grad is not None and torch.isfinite(p.grad).all().item() and p.grad.abs().max().item() > 0 for p in params):
                    raise RuntimeError("Missing finite nonzero intervention gradients")
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                engine.sync()
                elapsed += time.perf_counter() - start
                losses.append(objective.item())
                steps += 1
                if steps in (16, 64, 256):
                    save_checkpoint(engine, f"{args.method}_exposure{steps}", args.method,
                                    {"data_exposure": steps, "training_seconds": elapsed,
                                     "trainable_parameters": count, "learning_rate": best["learning_rate"],
                                     "comparison": "exact data exposure; supplementary checkpoint"})
        validation_loss = validate(engine, validation)
        info = {"level": level, "data_exposure": steps, "training_budget_seconds": budget,
                "budget_overshoot_seconds": elapsed - budget, "budget_policy": "finish one in-flight optimizer step",
                "training_seconds": elapsed, "seconds_per_step": elapsed / steps, "trainable_parameters": count,
                "hyperparameter_search_seconds": search_seconds, "hyperparameter_search_budget_seconds": 60,
                "search": candidates, "learning_rate": best["learning_rate"],
                "validation_response_nll": validation_loss, "training_losses": losses,
                "optimizer": "AdamW", "batch_size": 1, "maximum_prompt_tokens": 256, "maximum_response_tokens": 128,
                **mem.result(), "model": metadata(args)}
        name = f"{args.method}_{level}"
        save_checkpoint(engine, name, args.method, info)
        print(json.dumps({"event": "checkpoint", "variant": name, **{k:v for k,v in info.items() if k != "training_losses"}}), flush=True)
        if args.smoke:
            break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("method", choices=["dom", "learned", "lora"])
    p.add_argument("--smoke", action="store_true", help="only the first checkpoint; uses real train/validation data")
    args = p.parse_args()
    args.variant, args.device, args.dtype, args.seed, args.max_new_tokens = "baseline", "mps", "bfloat16", SEED, 1024
    split = prepare_split()
    engine = Engine(args)
    train(engine, args, split)


if __name__ == "__main__":
    main()
