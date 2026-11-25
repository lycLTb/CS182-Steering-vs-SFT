# Open questions (not yet answered)

- Layer index: learned steering says "layer 11, full-attention, same as DoM"; full-attention layers are 0-based [3,7,11,15,19,23], but DoM hooks `layers[10]` (a GDN layer). Same layer or not? The |cos| ≤ 0.03 vs DoM claim depends on it.
- Send issues to teammates first (HarmBench/WildJailbreak base mismatch → Jia; layer index → Jia & Suyang)?
- FLOPs shared assumptions (left as is): what N includes (embedding / lm_head counted?); which tokens are counted (padding, truncation, lm_head on response tokens only).
- Learned-steering "|cos| ≤ 0.03 vs DoM" result left out of the draft until the layer-index question is resolved.
