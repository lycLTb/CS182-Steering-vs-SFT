# Yanchi — baseline & evaluation pipeline

Round-1 baseline for `Qwen/Qwen3.5-0.8B` and the evaluation pipeline every model (base / SFT / LoRA) goes through.
Self-contained folder: all paths are relative to `yanchi/`, nothing outside it is imported.

## Layout
```
yanchi/
├── core/                    # shared code
│   ├── paths.py             # ROOT, DATA, MODELS, OUTPUTS, BASE_DIR
│   ├── model.py             # load_model, resolve_model, model_info, render, generate_batch, stop tokens
│   ├── io.py                # read_jsonl, append_jsonl, done_ids
│   ├── judge.py             # judge prompts (verbatim) + parsers: HarmBench / XSTest / MT-Bench
│   └── deepseek.py          # judge API config; key from env / .deepseek_api_key (never in code)
├── scripts/
│   ├── prepare_data.py      # download benchmarks, build ID lists          ┐
│   ├── export_base.py       # pinned base model -> models/                 │ pipeline
│   ├── generate.py          # responses for safety benchmarks + MT-Bench   │
│   ├── run_mmlu.py          # MMLU 5-shot (lm-eval)                         ┘
│   ├── smoke_test.py        # env / speed check                             ┐
│   ├── judge_smoketest.py   # does DeepSeek's filter block judge inputs?    │ experiments (kept as record)
│   └── judge_xstest_ablation.py  # XSTest judge position bias               ┘
├── data/ids/                # fixed ID lists (committed); data/*.jsonl git-ignored
├── models/                  # git-ignored model dirs (save_pretrained format)
└── outputs/<run>/           # *.json summaries committed; *.jsonl (responses, judge raw) git-ignored
```

## Quick start (`conda activate cs182`)
Scripts run from any directory (they add `yanchi/` to `sys.path`); `--out` / `--model` are relative to `yanchi/`.
```bash
python scripts/prepare_data.py                      # needs HF access to allenai/wildjailbreak (gated)
python scripts/export_base.py                       # -> models/qwen3.5-0.8b-base (1.5 GB)
python scripts/generate.py --bench all --out outputs/baseline   # ~20 min on a 4070 SUPER, resumable
python scripts/run_mmlu.py --out outputs/baseline               # ~4 min
export DEEPSEEK_API_KEY=...                          # or: echo KEY > .deepseek_api_key
```
## Environment
- Python 3.12, torch 2.10.0+cu128, transformers 5.23, flash-linear-attention 0.6.0, lm_eval 0.4.13, peft, openai.
  Full pins: `requirements.lock.txt`, `environment.yml`.
- Gated DeltaNet uses FLA Triton kernels (verified). **Don't install `causal-conv1d`**: transformers then uses its
  torch conv1d fallback — same speed for this model, but the CUDA kernel gives *different* greedy outputs in bf16
  (only 6/64 identical sequences at bs 64), so mixing environments would add noise. `model_info` records
  `causal_conv1d_installed` in every result file.

## Data (`scripts/prepare_data.py [step ...]`, seed 182)
| File | Count |
|---|---|
| harmbench_test / harmbench_val | 159 / 41 (standard behaviours, official CSV; val = tuning only) |
| wj_eval_adv_harmful / wj_eval_adv_benign | 2000 / 210 (full eval split); `train` (261,559) cached in HF cache |
| xstest | 250 safe + 200 unsafe |
| mtbench_questions / reference / judge_prompts | 80 / 30 / 8 (FastChat) |
| MMLU | full `validation` split (1531 Qs), few-shot from `dev`. (`mmlu_subset_ids.json` = old 50/subject test subset, unused) |

## Models: one local format for base and fine-tuned models
- The HF repo is multimodal (`Qwen3_5ForConditionalGeneration`); `export_base.py` saves only the text model
  (`Qwen3_5ForCausalLM`, 752M params, bf16 → lossless). Weights / tokenizer / chat template verified bit-identical to HF.
- Pinned revision `2fc0636…`; `provenance.json` records source, revision, `weights_sha256`.
- Every evaluated model is such a dir; scripts take `--model <dir>` (default = exported base). Result files record
  `model`, `weights_sha256`, `provenance`.
- **Stop tokens:** the HF repo has no `generation_config.json`; the derived default stops only at `<|endoftext|>`.
  The base model emits `<|im_end|> \n <|endoftext|>`, but a fine-tuned model may not. `generate_batch` always passes
  `eos_token_id=[<|im_end|>, <|endoftext|>]`; the exported `generation_config.json` sets the same.
  Verified: regenerating XSTest with the local model + this change gives **450/450 identical responses** to
  `outputs/baseline`; `n_new_tokens` is exactly 2 lower for the 269 `stop` rows (baseline counts are +2), unchanged
  for the 181 `length` rows.

## Handoff: SFT / LoRA models
1. **Start from** `models/qwen3.5-0.8b-base`.
2. **Format** — train on exactly what eval renders:
   ```python
   from core.model import render
   prompt = render(tok, [{"role": "user", "content": x}])   # ends with "<think>\n\n</think>\n\n"
   text   = prompt + response + "<|im_end|>"                # loss on response + <|im_end|> only
   ```
   (Single-turn: equals `apply_chat_template(user+assistant, enable_thinking=False)` up to a trailing `\n`.
   Multi-turn: build per turn — the template renders *earlier* assistant turns without the empty think block.)
3. **LoRA target modules** (only 6 of 24 layers have standard attention):
   | module | layers | shape (out, in) |
   |---|---|---|
   | `self_attn.{q,k,v,o}_proj` | 6 full-attention | q 4096×1024, k/v 512×1024, o 1024×2048 |
   | `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | 18 Gated DeltaNet | 6144×1024, 2048×1024, 1024×2048 |
   | `linear_attn.in_proj_{a,b}` | 18 | 16×1024 (gates — probably skip) |
   | `mlp.{gate,up,down}_proj` | all 24 | 3584×1024 / 1024×3584 |
   Embeddings are tied (`lm_head` = `embed_tokens`) — don't LoRA `lm_head`.
4. **Save merged** (`merge_and_unload()` → `save_pretrained` + tokenizer) to `models/<method>-<budget>/` with a
   `provenance.json` (method, base sha, data IDs, hyper-params, training FLOPs). Always evaluate merged models.
5. **Evaluate**: same commands, only `--model` changes:
   ```bash
   python scripts/generate.py --model models/<name> --bench all --out outputs/<name>
   python scripts/run_mmlu.py --model models/<name> --out outputs/<name>
   ```

## Generation & MMLU notes
- Greedy, bf16, `max_new_tokens=512`, non-thinking chat template; prompts sorted by length (desc), left-padded, bs 64.
- **Left padding is correct for Gated DeltaNet** (fp32 padded-vs-single max |Δlogit| = 0.008). But in bf16 the batch
  shape changes greedy outputs (3/16 identical bs=1 vs batched; divergences are paraphrases)
  → **keep `--bs 64` and the same ID lists for every method**.
- MT-Bench turn 2 is conditioned on the model's own turn-1 answer.
- MMLU: lm-eval materialises full-sequence logits (248k vocab, fp32) → `--bs 2` (bs ≥ 4 OOMs on 12 GB). No chat template.

### Baseline results (`outputs/baseline`)
- MMLU validation (1531 Qs, 5-shot): **48.92%** micro / 50.17% macro.
- Truncated at 512 tokens: HarmBench 48%, WJ harmful 66%, WJ benign 77%, XSTest 40%, MT-Bench turns 52%.
- ~10% of responses degenerate into repetition loops (greedy, small model).

## Judge (DeepSeek, `core/judge.py`, `core/deepseek.py`)
- Model `deepseek-flash` (= V4.1 Flash; snapshot not selectable — track `system_fingerprint`, so far `bc19ba97…`).
- Prompts verbatim: HarmBench classifier (HF model card), XSTest GPT-4 prompt (repo code, incl. its whitespace),
  MT-Bench single-answer prompts (FastChat; references for math/reasoning/coding).
- `judge_smoketest.py` (29 requests incl. clearly harmful content): **no blocks** — 0 `content_filter`, 0 HTTP 400.
- `judge_xstest_ablation.py` (30 XSTest items × 4 variants): with **thinking disabled the refusal judge mostly picks
  the last-listed class** (original order: 20/30 `partial_refusal`; classes reordered: 20/30 `full_compliance`;
  whitespace irrelevant). **Thinking enabled (`reasoning_effort=low`): 24/25 vs expected**, misses defensible.
  → Use thinking mode for all judges (temperature is then ignored → measure self-consistency, store raw outputs).
- TODO before the full judging run: self-consistency of thinking mode; check HarmBench yes/no for the same bias;
  ~50 human labels per judge type.

## Speed / implementation notes
| batch | tok/s | peak GB |
|---|---|---|
| 4 | ~41 | 1.6 |
| 16 | ~336 | 1.9 |
| 64 | ~1280 | 3.2 |

- Decode is per-step-latency bound (~45–50 ms/step regardless of batch) → use large batches.
- causal-conv1d kernel vs torch fallback (tested 2026-11-20, script in git history): decode speed equal (±5%),
  prefill ~10–15% faster with the kernel; each deterministic, outputs differ between them → we use the fallback.
