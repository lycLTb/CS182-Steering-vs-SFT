"""Memory-efficient causal-LM loss for a 248k-vocab model on 12 GB.

HF's built-in loss materialises fp32 logits for every position (B x T x 248k -> OOM at bs 8 x ~1k tokens).
Here the lm_head + cross-entropy run only on supervised positions, in checkpointed chunks, so at most one
chunk of logits lives in memory. Same value as HF ForCausalLMLoss: token-level mean over all labels != -100
in the batch (shifted by one).
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_ce_sum(lm_head, h, y):
    return F.cross_entropy(lm_head(h).float(), y, reduction="sum")


def causal_lm_loss(causal_lm, input_ids, attention_mask, labels, chunk: int = 1024):
    """causal_lm: a *ForCausalLM module (for PEFT: peft_model.get_base_model()), LoRA layers already injected."""
    hidden = causal_lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h = hidden[:, :-1]
    y = labels[:, 1:]
    keep = y != -100
    h, y = h[keep], y[keep]  # (n_supervised, d), (n_supervised,)
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, y.numel(), chunk):
        total = total + checkpoint(_chunk_ce_sum, causal_lm.lm_head, h[i:i + chunk], y[i:i + chunk],
                                   use_reentrant=False)
    return total / y.numel()
