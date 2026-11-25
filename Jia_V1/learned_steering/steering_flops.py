#!/usr/bin/env python3
"""Hardware-independent training FLOPs for the steering/LoRA/DoM comparison.

Matmul FLOPs only: 2 x (parameters touched) per token, using the real per-layer
parameter counts read from the loaded model. Embedding lookups are free; the
tied LM head (d x V) is applied to response tokens only, exactly as
train_methods.response_loss does. Attention-score and DeltaNet-recurrence
terms are left out (a few percent at T <= 384); steering_flops_check.py
measures the gap with torch's FlopCounterMode.

Teammates can import ModelCosts so every method is placed on the same axis.
"""


class ModelCosts:
    def __init__(self, layer_params, hidden, vocab):
        self.layer_params = list(layer_params)
        self.hidden = hidden
        self.vocab = vocab
        self.head = hidden * vocab

    @classmethod
    def from_model(cls, model):
        base = model.get_base_model() if hasattr(model, "peft_config") else model
        layers = base.model.layers
        counts = [sum(p.numel() for p in layer.parameters()) for layer in layers]
        return cls(counts, base.config.hidden_size, base.lm_head.weight.shape[0])

    def body(self, start=0, stop=None):
        return sum(self.layer_params[start:stop])

    def forward(self, tokens, response_tokens):
        """One forward pass: all layers on every token, LM head on response tokens."""
        return 2 * self.body() * tokens + 2 * self.head * response_tokens

    def learned_step(self, layer, tokens, response_tokens):
        """Forward + backward for a vector added at the output of `layer`.

        Backward computes activation gradients only (weights are frozen) through
        the LM head and layers layer+1 .. last; nothing below the vector.
        """
        backward = 2 * self.body(layer + 1) * tokens + 2 * self.head * response_tokens
        return self.forward(tokens, response_tokens) + backward

    def lora_step(self, first_lora_layer, tokens, response_tokens, lora_parameters=0):
        """Forward + backward for LoRA whose lowest adapted layer is first_lora_layer."""
        forward = self.forward(tokens, response_tokens) + 2 * lora_parameters * tokens
        backward = 2 * self.body(first_lora_layer) * tokens + 2 * self.head * response_tokens
        weight_gradients = 2 * lora_parameters * tokens
        return forward + backward + weight_gradients

    def dom_extraction(self, prompt_tokens, last_layer=None):
        """Forward over the prompt only. last_layer=None means the full stack
        (what train_methods.py runs); last_layer=11 stops at the hook."""
        stop = None if last_layer is None else last_layer + 1
        return 2 * self.body(0, stop) * prompt_tokens

    def describe(self):
        return {"layer_parameters": self.layer_params, "body_parameters": self.body(),
                "lm_head_parameters": self.head, "hidden": self.hidden, "vocab": self.vocab,
                "formula": "forward = 2*N_body*T + 2*d*V*R; learned backward = 2*N_(layer+1..)*T + 2*d*V*R",
                "excluded": "attention scores, DeltaNet recurrence, norms, elementwise ops"}
