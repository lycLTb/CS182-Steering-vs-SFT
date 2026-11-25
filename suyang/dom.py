"""Fixed layer-11 difference-of-means steering for the shared Qwen baseline."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import torch

from shenghe_shao.soft_prompt import load_model as load_base_model

LAYER_NUMBER = 11
LAYER_INDEX = LAYER_NUMBER - 1
HARMFUL = {"vanilla_harmful", "adversarial_harmful"}
BENIGN = {"vanilla_benign", "adversarial_benign"}


def load_model(path, device):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return load_base_model(path, device=device, dtype=dtype)


def decoder_block(model):
    return model.model.layers[LAYER_INDEX]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def block_hook(model, hook):
    handle = decoder_block(model).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def hidden_states(output):
    return output[0] if isinstance(output, tuple) else output


@dataclass
class DoM:
    delta: torch.Tensor
    strength: float
    config_id: str

    @classmethod
    def load(cls, path, strength):
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("DoM strength must be finite and nonnegative")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["layer_number"] != LAYER_NUMBER:
            raise ValueError(f"DoM checkpoint must target decoder block {LAYER_NUMBER}")
        delta = checkpoint["delta"]
        if delta.ndim != 1 or not torch.isfinite(delta).all() or delta.norm() == 0:
            raise ValueError("DoM checkpoint must contain a finite, nonzero vector")
        return cls(delta, strength, f"dom_l11_{sha256(path)[:16]}_s{strength}")

    @contextmanager
    def apply(self, model):
        if self.strength == 0:
            yield
            return
        weight = model.get_input_embeddings().weight
        if self.delta.numel() != weight.shape[1]:
            raise ValueError("DoM vector width does not match the model")
        shift = (self.strength * self.delta).to(device=weight.device, dtype=weight.dtype)
        if not torch.isfinite(shift).all():
            raise ValueError("Scaled DoM vector is not finite in the model dtype")

        def add_shift(_module, _inputs, output):
            steered = hidden_states(output) + shift
            return (steered, *output[1:]) if isinstance(output, tuple) else steered

        with block_hook(model, add_shift):
            yield
