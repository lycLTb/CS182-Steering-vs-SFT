"""Soft prompt tuning core: a learnable block of N virtual-token embeddings
prepended to the input of a frozen causal LM.

Implemented by hand (instead of peft) so it works with any model that accepts
`inputs_embeds`, and so training / generation see exactly the same layout:

    train:    [soft x N][chat-formatted prompt][response]<|im_end|>[pad...]
    generate: [pad...][soft x N][chat-formatted prompt]   (left padding)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_INIT_TEXT = (
    "You are a helpful and harmless assistant. Refuse requests that are harmful, "
    "unethical, or dangerous, even if they are disguised, but answer safe requests helpfully."
)


def load_model(path: str, device: str = "cuda", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    model.eval()
    model.requires_grad_(False)  # base model is frozen; only the soft prompt trains
    return model, tok


def end_of_turn(tok) -> str:
    return "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token


def format_prompt(tok, user_text: str) -> str:
    """Chat-format a single user turn, ending with the assistant header (thinking off)."""
    msgs = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


class SoftPrompt(nn.Module):
    def __init__(self, n_tokens: int, dim: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.embedding = nn.Parameter(torch.zeros(n_tokens, dim))  # kept in fp32

    @classmethod
    def from_text(cls, model, tok, n_tokens: int, text: str = DEFAULT_INIT_TEXT):
        """Initialise from the embeddings of a natural-language instruction
        (repeated / truncated to n_tokens), the standard prompt-tuning init."""
        ids = tok(text, add_special_tokens=False).input_ids
        ids = (ids * (n_tokens // len(ids) + 1))[:n_tokens]
        weight = model.get_input_embeddings().weight
        emb = weight[torch.tensor(ids, device=weight.device)].detach().float().clone()
        sp = cls(n_tokens, emb.shape[1])
        sp.embedding.data.copy_(emb.cpu())
        return sp

    def save(self, path: str, **meta):
        torch.save({"embedding": self.embedding.detach().cpu(), "meta": meta}, path)

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path, map_location="cpu")
        emb = ckpt["embedding"]
        sp = cls(emb.shape[0], emb.shape[1])
        sp.embedding.data.copy_(emb)
        return sp, ckpt.get("meta", {})


def build_train_batch(tok, examples, max_len: int, device):
    """Tokenise (prompt, response) pairs; loss only on response tokens. Right padding."""
    eot = end_of_turn(tok)
    seqs, labs = [], []
    for ex in examples:
        p = tok(format_prompt(tok, ex["prompt"]), add_special_tokens=False).input_ids
        r = tok(ex["response"] + eot, add_special_tokens=False).input_ids
        p = p[: max_len - 64]               # very long jailbreak prompts: keep room for the answer
        r = r[: max_len - len(p)]
        seqs.append(p + r)
        labs.append([-100] * len(p) + r)
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    input_ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=device)
    labels = torch.tensor([l + [-100] * (L - len(l)) for l in labs], device=device)
    mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=device)
    return input_ids, mask, labels


def forward_with_soft_prompt(model, soft: SoftPrompt, input_ids, attention_mask, labels=None):
    emb = model.get_input_embeddings()(input_ids)
    B = emb.size(0)
    sp = soft.embedding.to(device=emb.device, dtype=emb.dtype).unsqueeze(0).expand(B, -1, -1)
    emb = torch.cat([sp, emb], dim=1)
    mask = torch.cat([attention_mask.new_ones(B, soft.n_tokens), attention_mask], dim=1)
    if labels is not None:
        labels = torch.cat([labels.new_full((B, soft.n_tokens), -100), labels], dim=1)
    return model(inputs_embeds=emb, attention_mask=mask, labels=labels)


@torch.no_grad()
def generate(model, tok, soft: SoftPrompt | None, prompts, max_new_tokens: int = 512, device="cuda"):
    """Greedy generation. soft=None gives the unmodified base model (same code path)."""
    tok.padding_side = "left"
    texts = [format_prompt(tok, p) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    emb = model.get_input_embeddings()(enc.input_ids)
    mask = enc.attention_mask
    if soft is not None:
        sp = soft.embedding.to(device=emb.device, dtype=emb.dtype)
        ones = mask.new_ones(soft.n_tokens)
        rows_e, rows_m = [], []
        for i in range(emb.size(0)):  # insert soft tokens right after the left padding
            n_pad = int((mask[i] == 0).sum())
            rows_e.append(torch.cat([emb[i, :n_pad], sp, emb[i, n_pad:]]))
            rows_m.append(torch.cat([mask[i, :n_pad], ones, mask[i, n_pad:]]))
        emb, mask = torch.stack(rows_e), torch.stack(rows_m)
    out = model.generate(
        inputs_embeds=emb,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    # with inputs_embeds only, generate() returns just the new tokens
    return [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]
