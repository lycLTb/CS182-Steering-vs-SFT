"""Analytic training-FLOP accounting ("model FLOPs", not hardware FLOPs).

Convention (shared by every method so budgets are comparable):
  * Only real tokens count (attention_mask == 1): prompt + response, padding excluded.
    Prompt tokens count although they carry no loss -- they still go through forward and backward.
  * Total = per-sequence cost summed over sequences, so batch size / grad accumulation do not matter.
  * Gradient-checkpointing recomputation is NOT counted (hardware FLOPs, implementation detail).
  * Optimizer-step FLOPs are ignored (O(trainable params) per step; tiny for LoRA / soft prompt).

Per sequence of L real tokens:
  dense   matmul weights N (all base params; the tied embedding counts once, as the lm_head matmul --
          the input lookup itself is free):
            forward 2·N·L;  backward: activation grads 2·N·L always, weight grads 2·N·L only if trained
            -> frozen base: 4·N·L          trainable (full FT): 6·N·L
  LoRA    adapter params N_lora are trained: 6·N_lora·L
  mixers  parameter-free sequence mixing, forward cost F, backward ≈ 2F (grads w.r.t. its inputs) -> 3F
            full attention (per layer): QK^T + AV = 2·2·L²·d_attn   (Chinchilla convention: no causal halving)
            Gated DeltaNet (per layer): ≈ 6·H·d_k·d_v per token     (state read-out + delta-rule update)

So for a frozen base (LoRA, soft prompt, learned steering): FLOPs(L) = 4·N·L + 6·N_train·L + 3·mixer(L).
Soft prompt: use L + n_soft_tokens.
"""


def mixer_fwd_flops(cfg, L: int) -> float:
    """Forward FLOPs of the parameter-free sequence mixers for one sequence of length L."""
    types = cfg.layer_types
    n_full = sum(t == "full_attention" for t in types)
    n_lin = sum(t == "linear_attention" for t in types)
    d_attn = cfg.num_attention_heads * cfg.head_dim
    full = n_full * 4 * L * L * d_attn
    lin = n_lin * 6 * L * cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim
    return full + lin


class FlopCounter:
    """Accumulates training FLOPs batch by batch.

    n_base:   number of base-model parameters (frozen or not), tied weights counted once
    n_train:  trainable parameters that are *extra* matmul weights on top of the base (LoRA adapters);
              0 for soft prompt / steering vectors (their own cost is negligible)
    base_trainable: True for full fine-tuning (base weights get weight grads too)
    """

    def __init__(self, cfg, n_base: int, n_train: int = 0, base_trainable: bool = False):
        self.cfg, self.n_base, self.n_train = cfg, n_base, n_train
        self.k_base = 6 if base_trainable else 4
        self.total = 0.0

    def seq_flops(self, L: int) -> float:
        return (self.k_base * self.n_base + 6 * self.n_train) * L + 3 * mixer_fwd_flops(self.cfg, L)

    def add(self, lengths) -> float:
        """lengths: real-token count of each sequence in the batch. Returns FLOPs of this batch."""
        f = sum(self.seq_flops(int(L)) for L in lengths)
        self.total += f
        return f

    def describe(self) -> dict:
        return {"convention": "model FLOPs on real tokens; 4N (frozen) / 6N (trained) per token + 3x mixer fwd; "
                              "no grad-ckpt recompute, no optimizer",
                "n_base": self.n_base, "n_train": self.n_train, "k_base": self.k_base}
