"""Train-only state supervision for the existing TopK SAE.

The encoder only sees a current activation vector.  Physical labels are targets,
never encoder inputs.  A zero auxiliary weight follows the original pure-SAE
optimization path, including its random initialization and batch permutations.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from temp_scripts.activation_baseline import sample_weights, write_csv
from temp_scripts.actor_model import parameter_fingerprint
from temp_scripts.data_access import DEV_ROLES
from temp_scripts.sae_model import ActivationStandardizer, SAEConfig, TopKSAE
from temp_scripts.sae_training import infer_sae, weighted_mse


HEAD_MODES = ("dense", "top1", "top2", "top4", "slots1", "slots2", "slots")


def _head_connection_count(mode):
    if mode == "slots":
        return 4
    if mode.startswith("top"):
        return int(mode[3:])
    if mode.startswith("slots"):
        return int(mode[5:])
    return None


@dataclass(frozen=True)
class AssistedConfig:
    sae: SAEConfig = field(default_factory=lambda: SAEConfig(width=64, top_k=16, preprocessing="global"))
    state_weight: float = .1
    targets: tuple = (0, 1, 2)
    regression_loss: str = "mse"
    classification_loss: str = "bce"
    head_mode: str = "dense"
    action_weight: float = 0.
    selection: str = "reconstruction"
    label_mode: str = "true"
    name: str = "state_assisted"

    def validate(self):
        self.sae.validate()
        for key in ("state_weight", "action_weight"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(key + " must be finite and nonnegative")
        if (not self.targets or len(set(self.targets)) != len(self.targets)
                or any(isinstance(t, bool) or not isinstance(t, int) or t not in (0, 1, 2) for t in self.targets)):
            raise ValueError("targets must be distinct indices from (0, 1, 2)")
        if self.regression_loss not in ("mse", "huber"):
            raise ValueError("Unknown regression loss")
        if self.classification_loss not in ("bce", "balanced", "entropy"):
            raise ValueError("Unknown classification loss")
        if self.head_mode not in HEAD_MODES:
            raise ValueError("Unknown state-head mode")
        if self.head_mode.startswith("slots") and self.sae.width < 3 * _head_connection_count(self.head_mode):
            raise ValueError(self.head_mode + " needs at least " + str(3 * _head_connection_count(self.head_mode)) + " latent coordinates")
        if self.selection not in ("reconstruction", "joint"):
            raise ValueError("Unknown checkpoint selection")
        if self.label_mode not in ("true", "trajectory_shuffle"):
            raise ValueError("Unknown label mode")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        return self


class AssistedTopKSAE(TopKSAE):
    """Original SAE plus a linear target head (two regressions, one logit)."""
    def __init__(self, input_dim=64, width=64, top_k=16, head_mode="dense", head_seed=0):
        super().__init__(input_dim, width, top_k)
        if head_mode not in HEAD_MODES:
            raise ValueError("Unknown state-head mode")
        connections = _head_connection_count(head_mode)
        if head_mode.startswith("slots") and width < 3 * connections:
            raise ValueError(head_mode + " needs at least " + str(3 * connections) + " latent coordinates")
        self.head_mode, self.head_seed = head_mode, int(head_seed)
        # No head initialization consumes the pure SAE's random number stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.head_seed + 104729)
            self.state_head = nn.Linear(width, 3)
        mask = torch.ones(3, width)
        if head_mode.startswith("slots"):
            mask.zero_()
            for target in range(3):
                mask[target, target * connections:(target + 1) * connections] = 1.
        self.register_buffer("head_mask", mask)

    def effective_head_mask(self):
        if not self.head_mode.startswith("top"):
            return self.head_mask
        indices = self.state_head.weight.detach().abs().topk(min(_head_connection_count(self.head_mode), self.width), dim=1).indices
        return torch.zeros_like(self.state_head.weight).scatter(1, indices, 1.)

    def predict_state(self, z):
        return F.linear(z, self.state_head.weight * self.effective_head_mask(), self.state_head.bias)


def fit_target_scaler(targets, weights, floor=1e-6):
    """Fit current-state target transforms using training labels only."""
    y = np.asarray(targets, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.ndim != 2 or y.shape[1] != 3 or not len(y) or w.shape != (len(y),):
        raise ValueError("Need nonempty N by 3 training targets and matching weights")
    if not np.isfinite(y).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Invalid targets or weights")
    if not np.isin(y[:, 2], (0., 1.)).all():
        raise ValueError("Contact target must be binary")
    mean = np.average(y[:, :2], axis=0, weights=w)
    std = np.sqrt(np.average((y[:, :2] - mean) ** 2, axis=0, weights=w))
    scale = np.where(std < floor, 1., std)
    prevalence = float(np.average(y[:, 2], weights=w))
    entropy = -(prevalence * math.log(prevalence) + (1 - prevalence) * math.log(1 - prevalence)) if 0 < prevalence < 1 else 1.
    return {"mean": mean.tolist(), "scale": scale.tolist(), "small_variance": (std < floor).tolist(),
            "contact_prevalence": prevalence,
            "contact_positive_weight": (1 - prevalence) / prevalence if 0 < prevalence < 1 else 1.,
            "contact_entropy": entropy, "contact_single_class": prevalence in (0., 1.)}


def transform_targets(targets, scaler):
    y = np.asarray(targets, dtype=float).copy()
    y[:, :2] = (y[:, :2] - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    return y.astype(np.float32)


def assign_training_labels(targets, splits, groups, label_mode="true", seed=0):
    """Break activation/label pairing by a train-only trajectory derangement.

    A destination trajectory receives one entire source trajectory's sequence.
    Different lengths use nearest-neighbour normalized frame positions. No
    validation rows, inputs, caller arrays, or batch-order RNG are changed.
    """
    y, splits, groups = np.asarray(targets), np.asarray(splits), np.asarray(groups)
    if y.ndim != 2 or y.shape[1] != 3 or splits.shape != (len(y),) or groups.shape != (len(y),):
        raise ValueError("Labels, splits and groups must align")
    if label_mode not in ("true", "trajectory_shuffle"):
        raise ValueError("Unknown label mode")
    train_indices = np.flatnonzero(splits == "train")
    source_indices = train_indices.copy()
    mapping = []
    permutation_seed = None
    if label_mode == "trajectory_shuffle":
        unique = np.unique(groups[train_indices])
        if len(unique) < 2:
            raise ValueError("Trajectory shuffle needs at least two training trajectories")
        permutation_seed = int(seed) + 32452843
        label_rng = np.random.default_rng(permutation_seed)
        order = np.arange(len(unique))
        # Exclude fixed points: no trajectory keeps its own state labels.
        for _ in range(100):
            permutation = label_rng.permutation(len(unique))
            if np.all(permutation != order):
                break
        else:
            permutation = np.roll(order, int(label_rng.integers(1, len(unique))))
        for destination_group, source_group in zip(unique, unique[permutation]):
            destination = np.flatnonzero((splits == "train") & (groups == destination_group))
            source = np.flatnonzero((splits == "train") & (groups == source_group))
            locations = np.rint(np.linspace(0, len(source) - 1, len(destination))).astype(int)
            source_indices[np.searchsorted(train_indices, destination)] = source[locations]
            mapping.append({"destination_trajectory": str(destination_group), "source_trajectory": str(source_group),
                "destination_frames": len(destination), "source_frames": len(source)})
        supervised = y.copy()
        supervised[train_indices] = y[source_indices]
    else:
        supervised = y
    digest = hashlib.sha256(np.asarray(source_indices, dtype="<i8").tobytes()).hexdigest()
    metadata = {"label_mode": label_mode, "permutation_seed": permutation_seed,
        "trajectory_mapping": mapping, "train_source_indices": source_indices.tolist(),
        "train_destination_indices": train_indices.tolist(), "train_source_indices_sha256": digest,
        "resampling": "nearest normalized frame position via numpy.rint(linspace(0, source_length-1, destination_length))",
        "validation_labels": "unchanged true labels", "holdout_eligible": label_mode == "true",
        "purpose": "method candidate" if label_mode == "true" else "negative control breaking state correspondence"}
    return supervised, metadata


def state_loss_rows(prediction, target, config, target_scaler):
    """Return per-sample mean selected-target loss plus three component losses."""
    regression = ((prediction[:, :2] - target[:, :2]).square() if config.regression_loss == "mse"
                  else F.huber_loss(prediction[:, :2], target[:, :2], reduction="none", delta=1.))
    positive_weight = target_scaler["contact_positive_weight"] if config.classification_loss == "balanced" else 1.
    contact = F.binary_cross_entropy_with_logits(prediction[:, 2], target[:, 2], reduction="none",
                pos_weight=prediction.new_tensor(positive_weight))
    if config.classification_loss == "entropy":
        contact = contact / max(target_scaler["contact_entropy"], 1e-6)
    components = torch.column_stack((regression, contact))
    return components[:, list(config.targets)].mean(dim=1), components


def sae_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(("encoder.", "decoder.")):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_assisted_checkpoint(path, model, standardizer, target_scaler, config, seed, epoch,
                             reconstruction_score, joint_score, selection, label_assignment=None):
    torch.save({"format_version": "state_assisted_v1", "spec": {
        "input_dim": model.input_dim, "width": model.width, "top_k": model.top_k,
        "head_mode": model.head_mode, "head_seed": model.head_seed},
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "standardizer": standardizer.state_dict(), "target_scaler": target_scaler,
        "configuration": asdict(config), "seed": int(seed), "best_epoch": int(epoch),
        "best_validation_mse": float(reconstruction_score), "validation_joint": float(joint_score),
        "selection": selection, "label_assignment": label_assignment}, path)


def load_assisted_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != "state_assisted_v1":
        raise ValueError("Unsupported assisted-SAE checkpoint format")
    # Loading a model should not alter a caller's random number stream either.
    with torch.random.fork_rng(devices=[]):
        model = AssistedTopKSAE(**payload["spec"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, ActivationStandardizer.from_state_dict(payload["standardizer"]), payload


def fit_assisted(activations, targets, splits, groups, config, seed, folder, *, actor=None,
                 layer="h1", weighting="trajectory", device="cpu", progress=print):
    """Fit train labels only; choose checkpoint on validation by a fixed rule.

    Both reconstruction-best and joint-best checkpoints are recorded along the
    same optimization path. Early stopping follows config.selection. Therefore
    their training budgets are shared within a run, but different selection
    rules can lead to different stopping epochs in separate runs.
    """
    config.validate()
    cfg, folder = config.sae, Path(folder)
    x, y, splits, groups = (np.asarray(activations), np.asarray(targets), np.asarray(splits), np.asarray(groups))
    if x.ndim != 2 or y.shape != (len(x), 3) or splits.shape != (len(x),) or groups.shape != (len(x),):
        raise ValueError("Activations, three targets, splits and groups must align")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y[:, 2], (0, 1)).all():
        raise ValueError("Nonfinite data or nonbinary contact")
    train, validation = splits == "train", splits == "validation"
    if not train.any() or not validation.any() or not set(splits).issubset(DEV_ROLES):
        raise ValueError("Assisted fitting needs train and validation, with no test rows")
    if layer not in ("h1", "h2"):
        raise ValueError("Unknown actor layer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Use a nonnegative integer seed")
    if not folder.is_dir():
        raise ValueError("Create the experiment folder before fitting")
    if config.action_weight and actor is None:
        raise ValueError("An action loss requires the frozen actor")
    if actor is not None and any(p.requires_grad or p.grad is not None for p in actor.parameters()):
        raise ValueError("Actor parameters must already be frozen without gradients")
    actor_before = parameter_fingerprint(actor) if actor is not None else None
    weights, validation_weights = sample_weights(groups[train], weighting), sample_weights(groups[validation], "trajectory")
    standardizer = ActivationStandardizer.fit(x[train], weights, cfg.scale_floor, cfg.preprocessing)
    supervised_targets, label_assignment = assign_training_labels(y, splits, groups, config.label_mode, seed)
    target_scaler = fit_target_scaler(supervised_targets[train], weights, cfg.scale_floor)
    normalized, normalized_targets = standardizer.transform(x), transform_targets(supervised_targets, target_scaler)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AssistedTopKSAE(x.shape[1], cfg.width, cfg.top_k, config.head_mode, seed).to(device)
    initial_hash = sae_fingerprint(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=0.)
    training_values = torch.as_tensor(normalized[train], device=device)
    training_targets = torch.as_tensor(normalized_targets[train], device=device)
    training_weights = torch.as_tensor(weights, device=device, dtype=torch.float32)
    act_mean = torch.as_tensor(standardizer.mean, device=device, dtype=torch.float32)
    act_scale = torch.as_tensor(standardizer.scale, device=device, dtype=torch.float32)
    original_actions, action_scale = None, 1.
    actor_forward = None
    if config.action_weight:
        actor.eval()
        if next(actor.parameters()).device != next(model.parameters()).device:
            raise ValueError("Actor and SAE must be on the same device")
        actor_forward = actor.from_h1 if layer == "h1" else actor.from_h2
        with torch.no_grad():
            original_actions = actor_forward(torch.as_tensor(x, dtype=torch.float32, device=device))
        action_np = original_actions.detach().cpu().numpy()
        mean_action = np.average(action_np[train], axis=0, weights=weights)
        action_scale = max(float(np.average(((action_np[train] - mean_action) ** 2).mean(axis=1), weights=weights)), 1e-8)
        if not math.isfinite(action_scale):
            raise ValueError("Nonfinite original actor output")
    training_actions = original_actions[train] if original_actions is not None else None
    history, best_rec, best_joint, best_epoch, best_score = [], float("inf"), float("inf"), 0, float("inf")
    stale = updates = 0
    batch_hash = hashlib.sha256()
    started = perf_counter()

    def evaluate(z, rec, mask, w):
        rec_loss = weighted_mse(rec[mask], normalized[mask], w)
        with torch.no_grad():
            prediction = model.predict_state(torch.as_tensor(z[mask], dtype=torch.float32, device=device))
            target = torch.as_tensor(normalized_targets[mask], dtype=torch.float32, device=device)
            losses, components = state_loss_rows(prediction, target, config, target_scaler)
        state_loss = float(np.average(losses.cpu().numpy(), weights=w))
        parts = np.average(components.cpu().numpy(), axis=0, weights=w)
        action_loss = 0.
        if config.action_weight:
            with torch.no_grad():
                reconstructed = torch.as_tensor(rec[mask], device=device) * act_scale + act_mean
                new_action = actor_forward(reconstructed)
                action_rows = ((new_action - original_actions[mask]).square().mean(dim=1) / action_scale).cpu().numpy()
            action_loss = float(np.average(action_rows, weights=w))
        return {"mse_standardized": rec_loss, "state_loss": state_loss,
                "distance_loss": float(parts[0]), "opening_loss": float(parts[1]), "contact_loss": float(parts[2]),
                "action_loss": action_loss,
                "joint_loss": rec_loss + config.state_weight * state_loss + config.action_weight * action_loss}

    for epoch in range(cfg.max_epochs + 1):
        if epoch:
            model.train()
            for start in range(0, len(training_values), cfg.batch_size):
                if start == 0:
                    order = rng.permutation(len(training_values))
                    batch_hash.update(np.asarray(order, dtype="<i8").tobytes())
                indices = torch.as_tensor(order[start:start + cfg.batch_size], device=device)
                optimizer.zero_grad(set_to_none=True)
                z_batch, predicted = model(training_values[indices])
                row_loss = (predicted - training_values[indices]).square().mean(dim=1)
                # Preserve the exact original operation ordering at lambda=0.
                loss = (row_loss * training_weights[indices]).mean()
                if config.state_weight:
                    state_rows, _ = state_loss_rows(model.predict_state(z_batch), training_targets[indices], config, target_scaler)
                    loss = loss + config.state_weight * (state_rows * training_weights[indices]).mean()
                if config.action_weight:
                    new_action = actor_forward(predicted * act_scale + act_mean)
                    action_rows = (new_action - training_actions[indices]).square().mean(dim=1) / action_scale
                    loss = loss + config.action_weight * (action_rows * training_weights[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite assisted training loss")
                loss.backward()
                model.project_decoder_gradient()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError("Nonfinite assisted SAE gradient")
                optimizer.step()
                model.normalize_decoder()
                updates += 1
        z, rec = infer_sae(model, normalized, cfg.batch_size)
        tr = evaluate(z, rec, train, weights)
        val = evaluate(z, rec, validation, validation_weights)
        rec_improved, joint_improved = val["mse_standardized"] < best_rec, val["joint_loss"] < best_joint
        if rec_improved:
            best_rec = val["mse_standardized"]
            save_assisted_checkpoint(folder / "best_reconstruction.pt", model, standardizer, target_scaler,
                config, seed, epoch, val["mse_standardized"], val["joint_loss"], "reconstruction", label_assignment)
        if joint_improved:
            best_joint = val["joint_loss"]
            save_assisted_checkpoint(folder / "best_joint.pt", model, standardizer, target_scaler,
                config, seed, epoch, val["mse_standardized"], val["joint_loss"], "joint", label_assignment)
        score = val["mse_standardized"] if config.selection == "reconstruction" else val["joint_loss"]
        improved = score < best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
        elif epoch:
            stale += 1
        history.append({"epoch": epoch, "optimizer_steps": updates,
            **{"train_" + k: v for k, v in tr.items()}, **{"validation_" + k: v for k, v in val.items()},
            "best_validation_mse": best_rec, "best_validation_joint": best_joint,
            "is_new_best": improved, "batch_order_sha256": batch_hash.hexdigest()})
        if epoch == 0 or epoch % 10 == 0 or stale >= cfg.patience or epoch == cfg.max_epochs:
            write_csv(folder / "training_history.csv", history)
            progress(f"  epoch {epoch}/{cfg.max_epochs}: rec={val['mse_standardized']:.6g}, "
                     f"state={val['state_loss']:.6g}, joint={val['joint_loss']:.6g}; best={best_epoch}")
        if stale >= cfg.patience:
            break
    write_csv(folder / "training_history.csv", history)
    shutil.copyfile(folder / ("best_reconstruction.pt" if config.selection == "reconstruction" else "best_joint.pt"),
                    folder / "best_model.pt")
    model, standardizer, metadata = load_assisted_checkpoint(folder / "best_model.pt", device)
    z, normalized_reconstruction = infer_sae(model, normalized, cfg.batch_size)
    reconstructed = standardizer.inverse_transform(normalized_reconstruction)
    if actor is not None and (parameter_fingerprint(actor) != actor_before
                             or any(p.requires_grad or p.grad is not None for p in actor.parameters())):
        raise RuntimeError("Frozen actor changed or received gradients")
    result = {"seed": seed, "best_epoch": best_epoch, "epochs_completed": epoch,
        "optimizer_steps": updates, "early_stopped": stale >= cfg.patience,
        "best_validation_mse_standardized": float(metadata["best_validation_mse"]),
        "minimum_validation_mse_standardized": best_rec,
        "best_validation_joint": float(metadata["validation_joint"]), "best_selection_score": best_score,
        "best_is_initial_model": best_epoch == 0, "fitting_seconds": perf_counter() - started,
        "small_variance_coordinates": np.flatnonzero(standardizer.small_variance).tolist(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "selection": config.selection, "target_scaler": target_scaler,
        "initial_sae_sha256": initial_hash, "final_sae_sha256": sae_fingerprint(model),
        "batch_order_sha256": batch_hash.hexdigest(), "action_loss_train_variance": action_scale,
        "head_mode": config.head_mode, "head_nonzero_connections": model.effective_head_mask().sum(dim=1).int().tolist(),
        "label_mode": config.label_mode, "label_assignment": label_assignment,
        "actor_weights_unchanged": actor is None or parameter_fingerprint(actor) == actor_before,
        "encoder_inputs": "current activation only", "test_rows_used": 0}
    return model, standardizer, normalized, z, reconstructed, result
