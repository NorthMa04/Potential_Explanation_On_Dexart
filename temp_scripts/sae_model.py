"""Small TopK SAE and reversible, train-only activation preprocessing."""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SAEConfig:
    width: int = 128
    top_k: int = 8
    learning_rate: float = 0.001
    batch_size: int = 128
    max_epochs: int = 500
    patience: int = 50
    scale_floor: float = 1e-6

    def validate(self):
        for name in ("width", "top_k", "batch_size", "max_epochs", "patience"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if self.top_k > self.width:
            raise ValueError("top_k cannot exceed dictionary width")
        for name in ("learning_rate", "scale_floor"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        return self


@dataclass
class ActivationStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    small_variance: np.ndarray

    @classmethod
    def fit(cls, training_activations, weights, floor=1e-6):
        x, w = np.asarray(training_activations, dtype=float), np.asarray(weights, dtype=float)
        if x.ndim != 2 or len(x) == 0 or w.shape != (len(x),):
            raise ValueError("Preprocessing requires nonempty training rows and matching weights")
        if not np.isfinite(x).all() or not np.isfinite(w).all() or (w <= 0).any():
            raise ValueError("Invalid training activations or weights")
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("scale floor must be finite and positive")
        mean = np.average(x, axis=0, weights=w)
        std = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
        small = std < floor
        # Constant / nearly constant coordinates remain present, without amplification.
        return cls(mean, np.where(small, 1.0, std), small)

    def transform(self, x):
        return ((np.asarray(x, dtype=float) - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=float) * self.scale + self.mean

    def state_dict(self):
        return {"mean": torch.tensor(self.mean), "scale": torch.tensor(self.scale),
                "small_variance": torch.tensor(self.small_variance)}

    @classmethod
    def from_state_dict(cls, state):
        return cls(*(state[k].detach().cpu().numpy().copy() for k in ("mean", "scale", "small_variance")))


class TopKSAE(nn.Module):
    def __init__(self, input_dim=64, width=128, top_k=8):
        super().__init__()
        if not 1 <= top_k <= width or input_dim < 1:
            raise ValueError("Invalid input dimension, dictionary width or TopK")
        self.input_dim, self.width, self.top_k = input_dim, width, top_k
        self.encoder = nn.Linear(input_dim, width)
        self.decoder = nn.Linear(width, input_dim)
        with torch.no_grad():
            nn.init.normal_(self.decoder.weight)
            self.normalize_decoder()
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()
            self.decoder.bias.zero_()

    def encode(self, x):
        values, indices = torch.topk(self.encoder(x), k=self.top_k, dim=-1)
        return torch.zeros((*x.shape[:-1], self.width), device=x.device, dtype=x.dtype).scatter(
            -1, indices, torch.relu(values))

    def forward(self, x):
        z = self.encode(x)
        return z, self.decoder(z)

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))

    @torch.no_grad()
    def project_decoder_gradient(self):
        grad = self.decoder.weight.grad
        if grad is not None:
            grad.sub_(self.decoder.weight * (grad * self.decoder.weight).sum(dim=0, keepdim=True))


def save_checkpoint(path, model, standardizer, config, seed, best_epoch, best_score):
    torch.save({"format_version": 1, "spec": {"input_dim": model.input_dim,
                "width": model.width, "top_k": model.top_k},
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "standardizer": standardizer.state_dict(), "configuration": asdict(config),
                "seed": int(seed), "best_epoch": int(best_epoch),
                "best_validation_mse": float(best_score)}, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported SAE checkpoint format")
    model = TopKSAE(**payload["spec"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, ActivationStandardizer.from_state_dict(payload["standardizer"]), payload
