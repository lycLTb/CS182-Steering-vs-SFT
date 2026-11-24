# Progress notes (as of 2026-11-24)

Condensed record of decisions and findings from setting up the baseline. How-to details live in `../README.md`.

## Status
- **Done:** env (`cs182` conda), data download + fixed ID lists, base model export, full baseline generation
  (HarmBench test, WildJailbreak eval, XSTest, MT-Bench) and MMLU, DeepSeek judge prompts + two judge tests.
- **Not done:** full API judging run, judge validation (human labels), SFT / steering, moving `yanchi/` into the
  teammate's GitHub repo (plan: copy the folder in as-is, no history merge; code stays independent).
- Local-only (git-ignored, regenerate with scripts): `data/*.jsonl`, `models/`, `outputs/**/*.jsonl`.

## Decisions (and why)
| Decision | Reason |
|---|---|
| Model `Qwen/Qwen3.5-0.8B`, revision `2fc0636…`, exported to `models/qwen3.5-0.8b-base` (text-only, bf16) | One local format for base and fine-tuned models; pinned weights. Export is bit-identical to HF. |
| Greedy, bf16, non-thinking template, `max_new_tokens=512`, **bs 64**, prompts sorted by length | bf16 batch shape changes greedy outputs → every method must use the same bs and ID lists. |
| Stop tokens passed explicitly: `<|im_end|>`, `<|endoftext|>` | HF repo has no generation_config; default stops only at `<|endoftext|>`. Fine-tuned models might never emit it. Baseline `n_new_tokens` for `stop` rows are +2 (text identical). |
| **Do not install `causal-conv1d`** (torch fallback) | Same speed for this model, but the CUDA kernel changes bf16 greedy outputs. `causal_conv1d_installed` is recorded in result files. |
| WildJailbreak: full eval split (2000 adv-harmful + 210 adv-benign) | User choice (doc originally said 500 subset). |
| MMLU: full **validation** split (1531 Qs), 5-shot from dev, no chat template, lm-eval bs 2 | User choice (doc originally said 50/subject test subset). bs ≥ 4 OOMs (248k-vocab logits). |
| Compute budgets in **training FLOPs**, not GPU time | Methods run on different members' machines. Written into the proposal. FLOP counting method still open (2N fwd / ~4N fwd+bwd estimate, or `torch.utils.flop_counter`; check FLA kernels are counted). |
| Judge: DeepSeek `deepseek-flash` (V4.1 Flash), **thinking enabled** (`reasoning_effort=low`) | Non-thinking mode showed strong position bias (below). Snapshot can't be pinned → track `system_fingerprint` (`bc19ba97…` so far). |
| API key only via env `DEEPSEEK_API_KEY` or git-ignored `yanchi/.deepseek_api_key` | Folder will be pushed to a shared repo. |

## Baseline numbers (`outputs/baseline`)
- MMLU validation: **48.92%** micro / 50.17% macro. A teammate got 50.65 on a Mac — within sampling error
  (SE ≈ 1.3 pt for 1531 Qs); first check split / shots / aggregation / chat template before comparing.
- Keyword refusal rate (rough, not the metric): HarmBench 55%, WJ adv-harmful 31%, WJ adv-benign 16%,
  XSTest safe 4% / unsafe 67%.
- Truncated at 512 tokens: 40–77% depending on benchmark (MT-Bench turns 52% — may hurt MT-Bench scores;
  consider 1024 for MT-Bench only). ~10% of responses fall into repetition loops (greedy + small model).

## Judge findings
- **Provider filter is not a problem:** 29 requests incl. clearly harmful content → 0 `content_filter`, 0 HTTP 400.
- **HarmBench judge looks sensible:** "complies but not actually harmful" (disclaimers, nonsense steps, harmless
  fiction) → `no`. ASR ≠ 1 − refusal rate, as expected.
- **XSTest refusal judge, thinking off = position bias:** 30 items × 4 variants (`scripts/judge_xstest_ablation.py`):
  - original class order → 20/30 `partial_refusal` (the last-listed class); reordered → 20/30 `full_compliance`
    (now last). Removing the prompt's odd whitespace changes nothing.
  - thinking on → 24/25 agree with expected; the misses are defensible under XSTest definitions.
- **Next judge steps:** (1) self-consistency of thinking mode (temperature is ignored) — rerun 2×, majority-of-3
  if < ~95% agree; (2) test HarmBench yes/no for the same bias; (3) ~50 human labels per judge type;
  (4) then the full run (~3M tokens, cheap). Update `doc/experiment_setup.md` (temperature 0 / fixed snapshot no
  longer hold).

## Gotchas
- **WSL instability:** WSL crashed several times. Afterwards the GPU is sometimes unavailable
  (`GPU access blocked by the operating system`; dmesg `dxgvmb_send_create_process failed -75`) → `wsl --shutdown`.
  `generate.py` is resumable for this reason. WSL disk is on D:; C: was nearly full (Cursor extension updates,
  leftover WSL `swap.vhdx` files). Windows-side notes: `D:\wsl_c_drive_investigation.md`.
- WSL Ubuntu 20.04 has glibc 2.31 → prebuilt CUDA-extension wheels needing ≥ 2.32 fail at import (build from source).
- The HF repo is multimodal (`Qwen3_5ForConditionalGeneration`); `AutoModelForCausalLM` loads the text model only.
- Architecture: 24 layers = 18 Gated DeltaNet + 6 full attention; tied embeddings. LoRA target names are in the README.
  Left padding is correct for the recurrent layers (fp32 check).
- SFT data must be rendered exactly like eval: `render(tok, [user]) + response + "<|im_end|>"`. Earlier assistant
  turns in multi-turn history are rendered without the empty think block.
- lm-eval's `mmlu_*` tasks default to the test split; `run_mmlu.py` overrides `test_split` to validation.
- The environment's openai SDK (3.x) uses `httpx2`, not `httpx`.
