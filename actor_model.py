"""Restore the frozen DexArt actor tail without loading SB3, PointNet or simulation."""
from contextlib import AbstractContextManager
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn

CHECKPOINT_MEMBER = "assets/rl_checkpoints/laptop/laptop_nopretrain_0.zip"
PARAMETER_SHAPES = {
    "mlp_extractor.policy_net.0.weight": (64, 320),
    "mlp_extractor.policy_net.0.bias": (64,),
    "mlp_extractor.policy_net.2.weight": (64, 64),
    "mlp_extractor.policy_net.2.bias": (64,),
    "action_net.weight": (22, 64), "action_net.bias": (22,),
}


class FrozenActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_extractor = nn.Module()
        self.mlp_extractor.policy_net = nn.Sequential(nn.Linear(320, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.action_net = nn.Linear(64, 22)

    def forward(self, features):
        return self.action_net(self.mlp_extractor.policy_net(features))

    def from_h1(self, h1):
        layers = self.mlp_extractor.policy_net
        return self.action_net(layers[3](layers[2](h1)))

    def from_h2(self, h2):
        return self.action_net(h2)


def load_actor(source, expected_sha256, device="cpu"):
    """Read only the needed ZIP members; never unpickle the SB3 data configuration."""
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError("找不到原权重资源包：%s；可用 --checkpoint 指定 dexart_assets.zip 或 laptop_nopretrain_0.zip。" % source)
    with zipfile.ZipFile(source) as archive:
        if "policy.pth" in archive.namelist():
            checkpoint = source.read_bytes()
            member = None
        elif CHECKPOINT_MEMBER in archive.namelist():
            checkpoint = archive.read(CHECKPOINT_MEMBER)
            member = CHECKPOINT_MEMBER
        else:
            raise ValueError("ZIP 中没有指定 laptop 原策略检查点")
    sha = hashlib.sha256(checkpoint).hexdigest()
    if sha != expected_sha256:
        raise ValueError("检查点 SHA256 与采集日志不一致；停止前向校验。")
    with zipfile.ZipFile(io.BytesIO(checkpoint)) as archive:
        config = json.loads(archive.read("data"))["policy_kwargs"]
        if config.get("net_arch") != [{"pi": [64, 64], "vf": [64, 64]}] or "ReLU" not in str(config.get("activation_fn")):
            raise ValueError("检查点网络配置与预期 actor 不符")
        state = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
    selected = {key: value for key, value in state.items() if key.startswith(("mlp_extractor.policy_net.", "action_net."))}
    if set(selected) != set(PARAMETER_SHAPES):
        raise ValueError("缺少或出现未预期的 actor 参数；不允许随机初始化补齐")
    for key, shape in PARAMETER_SHAPES.items():
        value = selected[key]
        if tuple(value.shape) != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Invalid actor parameter: " + key)
    actor = FrozenActor()
    actor.load_state_dict(selected, strict=True)
    actor.requires_grad_(False)
    actor.eval()
    actor.to(device)
    metadata = {"source": str(source), "member": member, "checkpoint_sha256": sha,
                "load_mode": "torch.load(weights_only=True), six actor tensors, strict state dict",
                "parameter_shapes": {key: list(shape) for key, shape in PARAMETER_SHAPES.items()},
                "parameters": sum(v.numel() for v in selected.values()), "dtype": "float32", "device": str(device),
                "feature_extractor_restored": False, "optimizer_created": False, "parameters_trainable": False}
    return actor, metadata


def parameter_fingerprint(actor):
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class HookRecorder(AbstractContextManager):
    """Observe original post-ReLU/module outputs; callbacks always return None."""
    def __init__(self, actor):
        self.values = {name: [] for name in ["h1", "h2", "mu"]}
        self.handles = []
        self.modules = {"h1": actor.mlp_extractor.policy_net[1], "h2": actor.mlp_extractor.policy_net[3], "mu": actor.action_net}

    def __enter__(self):
        for name, module in self.modules.items():
            def capture(_module, _inputs, output, label=name):
                self.values[label].append(output.detach().cpu().clone().numpy())
                return None
            self.handles.append(module.register_forward_hook(capture))
        return self

    def reset(self):
        for values in self.values.values():
            values.clear()

    def check(self, expected, rows):
        for name, values in self.values.items():
            count = 1 if name in expected else 0
            if len(values) != count:
                raise RuntimeError("Unexpected hook count for %s: %d != %d" % (name, len(values), count))
            if values and values[0].shape != (rows, 22 if name == "mu" else 64):
                raise RuntimeError("Invalid hook batch shape: " + name)

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


def replay(actor, arrays, batch_size=256):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(actor.parameters()).device
    n = len(arrays["features_fused"])
    predictions, checks = {}, []
    stages = [("A", "activations_l2", actor.from_h2, ["mu"]),
              ("B", "activations", actor.from_h1, ["h2", "mu"]),
              ("C", "features_fused", actor, ["h1", "h2", "mu"])]
    for stage, key, forward, observed in stages:
        collected = {name: [] for name in observed}
        with HookRecorder(actor) as recorder:
            for start in range(0, n, batch_size):
                stop = min(n, start + batch_size)
                recorder.reset()
                x = torch.as_tensor(arrays[key][start:stop], dtype=torch.float32, device=device)
                with torch.no_grad():
                    output = forward(x)
                if output.requires_grad:
                    raise RuntimeError("Unexpected gradient tracking")
                recorder.check(observed, stop - start)
                for name in observed:
                    collected[name].append(recorder.values[name][0])
                checks.append({"stage": stage, "action_start": start, "action_stop": stop,
                               "hook_calls": {name: len(values) for name, values in recorder.values.items()}})
        for name, values in collected.items():
            predictions[stage + "_" + name] = np.concatenate(values)
        predictions[stage + "_action"] = np.clip(predictions[stage + "_mu"], -1, 1)
    # Independent forward without any hooks, under the same batching/device settings.
    baseline = []
    for start in range(0, n, batch_size):
        with torch.no_grad():
            x = torch.as_tensor(arrays["features_fused"][start:start + batch_size], dtype=torch.float32, device=device)
            baseline.append(actor(x).detach().cpu().numpy())
    baseline = np.concatenate(baseline)
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("Nonfinite replay output")
    checks_info = {"calls": checks, "hooks_removed": all(not m._forward_hooks for m in [actor.mlp_extractor.policy_net[1], actor.mlp_extractor.policy_net[3], actor.action_net]),
                   "hook_output_unchanged_exact": bool(np.array_equal(baseline, predictions["C_mu"])),
                   "max_hook_vs_no_hook_abs_error": float(np.max(np.abs(baseline.astype(float) - predictions["C_mu"].astype(float))))}
    predictions["D_action"] = np.clip(arrays["action_mean_raw"], -1, 1)
    return predictions, checks_info
