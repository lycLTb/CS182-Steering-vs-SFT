# Fixed layer-11 difference-of-means steering

Use the same Qwen3.5-0.8B checkpoint, WildJailbreak split files, and generation settings as the base run. The fitting script reads the prompt and class from each training row. It extracts the block output at the final chat-formatted prompt token and subtracts the pooled benign mean from the pooled harmful mean. During evaluation, it adds a nonnegative multiple of that vector to the output of decoder block 11 (`layers[10]`) during both prefill and decoding. Model weights stay frozen.

Run from the repository root with the module commands below. The DoM scripts reuse the team's existing Qwen loading and prompt formatting. Set `WORK` to the local parent of `cs182_assets`. Copy the exact `splits/` files used by the group into that directory. Point `model` at a local copy of the same Qwen3.5-0.8B checkpoint revision as the group.

The commands below use Apple Silicon MPS with FP32 and batch size 1. On NVIDIA use `device=cuda` for BF16. Use `device=cpu` for CPU FP32. Run your local baseline with the same device, dtype, batch size, and generation settings as DoM. When runs use different hardware, record the device and dtype with the timings.

```bash
device=mps
export PYTORCH_ENABLE_MPS_FALLBACK=1
assets="$WORK/cs182_assets"
model="$assets/models/Qwen3.5-0.8B"
direction="$WORK/runs/dom_l11/direction.pt"

python -m suyang.fit_dom --model "$model" --device "$device" --batch_size 1 \
  --train "$assets/splits/train.jsonl" --output "$direction"

for strength in 0 0.5 1 2; do
  python -m suyang.generate --model "$model" --device "$device" --batch_size 1 --dom "$direction" \
    --dom_strength "$strength" --input "$assets/splits/dev.jsonl" \
    --output "$WORK/outputs/dom_l11_dev_${strength}.jsonl"
done
```

Use strength 0 to check parity with the base run. Choose a strength using judged harmful compliance and benign over-refusal on dev. Freeze that strength before generating responses on the held-out jailbreak split and the shared safety and over-refusal benchmarks. Use the same judge as for the base model.

```bash
# Replace 1.0 with the strength selected on dev.
strength=1.0
python -m suyang.generate --model "$model" --device "$device" --batch_size 1 --dom "$direction" \
  --dom_strength "$strength" --input "$assets/splits/test_jailbreak.jsonl" \
  --output "$WORK/outputs/dom_l11_test.jsonl"

python -m suyang.eval_mmlu --model "$model" --device "$device" --batch_size 1 --dom "$direction" \
  --dom_strength "$strength" --mmlu_dir "$assets/data/mmlu" \
  --output "$WORK/outputs/dom_l11_mmlu.json"
```

The fit artifact records the train-file SHA-256, class counts by data type, extraction time, device, dtype, and the vector. Peak allocated CUDA memory is recorded on CUDA runs. DoM outputs include a config ID derived from the artifact and strength. Generation preserves input metadata and adds a prompt ID based on the exact prompt text when the input has no ID.
