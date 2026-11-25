# Report syllabus — LoRA SFT part (Yanchi)

Only decisions that are already settled. Open questions are not recorded here.

## Scope and format
- My part of the report: **LoRA SFT** only — its method, experiment setup, experiments, limitations,
  plus a literature-review addition on LoRA placement in DeltaNet / hybrid models.
  Soft prompt and steering are written by teammates.
- Draft in **Markdown** first; NeurIPS LaTeX later.
- Full-parameter SFT is not done: fp32 master weights + grads + AdamW states ≈ 12 GB > 12 GB VRAM.
  Mention in one sentence.

## Literature review (addition)
Topic: where to put LoRA in linear-attention / SSM / hybrid models (non-trivial for Gated DeltaNet).
- Borobia et al. 2026, *Where Should LoRA Go? Component-Type Placement in Hybrid Language Models*,
  arXiv:2604.22127 — same model (Qwen3.5-0.8B) and r=16, α=32, lr 2e-4.
- Galim et al., *Parameter-Efficient Fine-Tuning of State Space Models*, ICML 2025, arXiv:2410.09016.
- Transformer common practice: LoRA (Hu et al. 2022), QLoRA (Dettmers et al. 2023),
  Biderman et al. 2024 (*LoRA Learns Less and Forgets Less*), Thinking Machines 2025 (*LoRA Without Regret*).
- Verify arXiv IDs of any further papers before citing (MambaPEFT, S0 Tuning, etc.).

## Method (SFT is covered in class → keep the SFT objective brief)
- Model: Qwen3.5-0.8B, hybrid architecture: 24 layers = 18 Gated DeltaNet + 6 full attention; tied embeddings.
- LoRA: W' = W + (α/r)·BA, base frozen (bf16), adapters fp32; r = 16, α = 32.
- Final target modules: all-linear —
  `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `mlp.{gate,up,down}_proj`;
  excluded: `in_proj_{a,b}` (16-dim gates), `conv1d`, tied `embed_tokens`/`lm_head`. 10.2M trainable params.
- Loss: standard SFT loss (one line).
- Training data and its order: just say they are the same as for the soft prompt; no further detail.
- **Learning-rate schedule**: 20-step linear warmup, then constant → a checkpoint saved after N examples
  is equivalent to a run stopped at N. One run gives all budgets.
- **Checkpoints / budgets**: 64, 128, 256, 512, 1000, 2000, 4000 examples (+ base = 0).
  Multiples of 8 (bs). Small budgets added because ASR already plateaus at 1000.
  64 and 128 fall inside warmup (LR not yet at peak).
- **FLOPs**: same formula as the soft prompt, counted on real (non-padding) tokens, batch-size independent:
  `4·N·L + 6·N_lora·L + 3·mixer(L)` per sequence. Brief; note that the 6·N_lora term (~1%) is included, not ignored.

## Experiment setup (evaluation protocol lives here)
- Model generation in non-thinking mode; greedy, max_new_tokens 512.
- Paper style, not a technical report: no script or file names in the text.
- Tuning on `dev.jsonl` (400); final evaluation once on `test_harmbench` (200, not split),
  `test_jailbreak` (2210), `test_xstest` (450), MMLU, MT-Bench.
- Judge: DeepSeek (`deepseek-flash`), one prompt for refusal + harmful, thinking off.
- Metrics: ASR (harmful prompts judged harmful), over-refusal (benign prompts refused), MMLU accuracy, MT-Bench score.

## Experiments
- Hyperparameter tuning: **learning rate only**. Swept 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3 on dev.

## Target-module runs (dev, lr 2e-4, r 16)
Naming for the report (define at first use, following Borobia et al.):
- **softmax-only**: `self_attn.{q,k,v,o}_proj` of the 6 softmax (full) attention layers; 1.08M params.
- **attn+GDN**: softmax attention + Gated DeltaNet projections (previously called "attention-only"); 4.9M.
- **all-linear**: attn+GDN + `mlp.{gate,up,down}_proj`; 10.2M.

**Done** — softmax-only (run `lora_softmax_r16_lr2e-4`, budgets 32–2000), to address the inconsistency
with Borobia et al. 2026 (softmax-only best on Qwen3.5-0.8B). all-linear 128/512 also done
(`runs/lora_all_r16_lr2e-4_upto512`, same data order).

Dev results, ASR / overall over-refusal / MMLU / adversarial_benign refusal (%). Base: 19.5 / 13.0 / 47.9 / 20.

| budget | softmax-only | attn+GDN | all-linear |
|---|---|---|---|
| 32 | 16.5 / 16.0 / 47.0 / 25 | – | – |
| 64 | 13.5 / 16.5 / 47.4 / 27 | – | 7.0 / 46.0 / 46.5 / 72 |
| 128 | 7.0 / 35.0 / 45.3 / 59 | – | 0.0 / 51.0 / 48.8 / 90 |
| 256 | 5.0 / 38.5 / 37.7 / 70 | – | 3.5 / 38.0 / 51.6 / 64 |
| 512 | 4.0 / 41.0 / 46.3 / 60 | – | 1.5 / 42.0 / 51.3 / 60 |
| 1000 | 1.5 / 39.0 / 47.9 / 68 | 0.5 / 34.0 / 51.1 / 59 | 0.5 / 30.0 / 50.5 / 50 |
| 2000 | 0.5 / 31.5 / 48.9 / 55 | 0.5 / 23.0 / 50.3 / 39 | 0.0 / 17.0 / 46.9 / 30 |
| 4000 | – | 0.5 / 25.0 / 47.0 / 40 | 0.0 / 19.5 / 47.4 / 33 |

Observations:
- At 2000 ex: all-linear > attn+GDN > softmax-only on over-refusal (adv_benign 30 / 39 / 55), similar ASR.
  Borobia's finding does not transfer to our task/setup.
- softmax-only learns safety more slowly per budget (ASR 13.5 vs 7.0 at 64 ex).
- Over-refusal rises then falls with budget (all-linear adv_benign 90% at 128 ex → 30% at 2000 ex).
- softmax-only MMLU dips to 37.7 at 256 ex, recovers to 46.3 at 512 ex (single seed, cause unknown).
- Mean train loss (steps 10–250): softmax-only 1.423, attn+GDN 1.395, all-linear 1.388.
  s/step: 0.98 / 1.44 / 1.41.

Caveats to state in the report:
- Single seed; dev n = 100 per type (SE ≈ 5 pt).
- lr 2e-4 was selected on attn+GDN; softmax-only (1/5 the params) may prefer a higher lr, so it may be
  under-served by this comparison. (Borobia et al. also use 2e-4.)
- FLOPs: the 4N convention assumes activation gradients through all layers; softmax-only's first trainable
  layer is layer 3, so its true backward FLOPs are ~6% below the reported value.

## To run (planned, not started)
1. **Final learning-rate sweep**: the existing sweep used attention+DeltaNet targets, not the final
   configuration. Once target modules are settled, re-sweep the learning rate on the final configuration.
- Figures:
  1. Training loss curve.
  2. FLOPs (budget) → test metrics curve.
