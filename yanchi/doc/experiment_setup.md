# Experiment Setup (Round 1: Baseline)

## Model
- `Qwen/Qwen3.5-0.8B` (instruct), bf16, official chat template.
- Non-thinking mode (model default); same for all later steering / LoRA runs.
- Architecture: 24 layers = 6 × (3 Gated DeltaNet + 1 Gated Attention), hidden dim 1024.
  Only 6 layers use standard attention — LoRA target modules must be stated explicitly.
- Decoding: greedy (temperature 0), `max_new_tokens = 512`.

## Benchmarks

| Benchmark | Split used | Size (round 1) | Metric | Judge |
|---|---|---|---|---|
| HarmBench (text, standard, DirectRequest) | test | 159 (full) | ASR (harmful-content rate) | API, harmfulness prompt |
| WildJailbreak eval — adversarial harmful | eval | 500 (subset of 2000) | ASR | API, harmfulness prompt |
| WildJailbreak eval — adversarial benign | eval | 210 (full) | refusal rate | API, refusal prompt |
| XSTest v2 | test only | 250 safe + 200 unsafe (full) | refusal rate on safe / unsafe | API, refusal prompt |
| MMLU | test | ~2.8k (stratified, 50 per subject) | accuracy, 5-shot loglik | none |
| MT-Bench | — | 80 questions × 2 turns (full) | mean score 1–10 | API, official MT-Bench prompt |

Subsets are sampled once with a fixed seed and saved as ID lists; every method / budget uses the same IDs.

## Data roles (no leakage)
- **Training / steering construction:** WildJailbreak `train` (subset).
- **Hyperparameter tuning:** HarmBench standard `val` (41); optionally a held-out slice of WildJailbreak `train` adversarial-benign for over-refusal.
- **Final evaluation only:** HarmBench `test`, WildJailbreak `eval`, XSTest, MMLU `test`, MT-Bench.
- MMLU `dev` (5 per subject) is used only for few-shot exemplars.
- Never tune on XSTest or MT-Bench (test-only sets).

## Judging (external API)
- One judge model for all benchmarks; fixed dated snapshot, temperature 0. Model: TBD.
- Three prompts, not interchangeable:
  - **Harmfulness:** HarmBench classifier prompt (from `cais/HarmBench-Llama-2-13b-cls` model card), yes/no.
  - **Refusal:** XSTest definitions — full compliance / full refusal / partial refusal.
  - **Quality:** official MT-Bench single-answer grading prompt (with reference answers for math / reasoning / coding).
- ASR ≠ 1 − refusal rate (non-refusals can still be non-harmful); report separately.
- Store raw judge outputs; log and count API errors / moderation refusals.
- Validation: ~50 human-labeled samples per judgment type; report judge–human agreement.
- Numbers are not directly comparable to published HarmBench results (non-official judge); all internal comparisons use the same judge.

## Pipeline
1. Generate all responses with HF transformers (same code path later used with steering hooks), save as JSONL.
2. Judge offline in a separate pass.
3. MMLU via `lm-eval-harness` (`--tasks mmlu --num_fewshot 5`, subset via sample IDs).

## Environment notes
- Gated DeltaNet needs flash-linear-attention / causal-conv1d (Triton) kernels; native Windows may fall back to slow torch code. Prefer WSL2 on the RTX 4070 SUPER; run a speed smoke test first.
