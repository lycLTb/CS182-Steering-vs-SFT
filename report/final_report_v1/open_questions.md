# Open questions (not yet answered)

- Send issues to teammates first (HarmBench/WildJailbreak base mismatch → Jia; layer index → Jia & Suyang)?
- FLOPs shared assumptions (left as is): what N includes (embedding / lm_head counted?); which tokens are counted (padding, truncation, lm_head on response tokens only).
- RESOLVED layer index: DoM hooks the output of `layers[10]` (block 11, one-based); learned steering adds at the input of `layers[11]` (zero-based). Same residual-stream position. Pending: unify the wording in the paper; restore the |cos| ≤ 0.03 result?
