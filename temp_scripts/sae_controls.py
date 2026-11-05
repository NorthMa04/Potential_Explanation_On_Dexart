"""Train-only weighted PCA reconstruction controls for saved SAE windows."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from temp_scripts.actor_model import FrozenActor
from temp_scripts.activation_baseline import sample_weights
from temp_scripts.sae_model import ActivationStandardizer
from temp_scripts.sae_training import actor_outputs, check_original_actor


def pca_fidelity_controls(run, destination, ranks=(4, 8, 16, 24, 32)):
    run, destination = Path(run), Path(destination)
    config = json.loads((run / "configuration.json").read_text(encoding="utf-8"))
    with np.load(run / "selected_arrays.npz", allow_pickle=False) as data:
        h, groups, splits = (data[k] for k in ("h", "trajectory_id", "split"))
    if not set(splits).issubset({"train", "validation"}):
        raise ValueError("PCA controls must not fit or inspect test numerical rows")
    with np.load(run / "action_references.npz", allow_pickle=False) as data:
        original_mu, saved_mean, saved_action = (data[k] for k in ("original_mean_replayed", "saved_mean", "saved_action"))
    actor = FrozenActor().eval().requires_grad_(False)
    actor.load_state_dict(torch.load(run / "model" / "frozen_actor_state.pt", weights_only=True), strict=True)
    check_original_actor(actor_outputs(actor, h, config["layer"]), saved_mean, saved_action)
    train, validation = splits == "train", splits == "validation"
    weights = sample_weights(groups[train], config["weighting"])
    validation_weights = sample_weights(groups[validation])
    rows = []
    with threadpool_limits(1):
        for mode in ("coordinate", "global"):
            scaler = ActivationStandardizer.fit(h[train], weights, mode=mode)
            normalized = scaler.transform(h).astype(float)
            center = np.average(normalized[train], axis=0, weights=weights)
            _, _, basis = np.linalg.svd((normalized[train] - center) * np.sqrt(weights[:, None]), full_matrices=False)
            for rank in ranks:
                if not 1 <= rank <= len(basis):
                    raise ValueError("PCA rank exceeds the available training basis")
                reconstructed = scaler.inverse_transform((normalized - center) @ basis[:rank].T @ basis[:rank] + center)
                new_mu = actor_outputs(actor, reconstructed, config["layer"])
                error = np.clip(new_mu[validation], -1, 1) - np.clip(original_mu[validation], -1, 1)
                mse = np.average(((reconstructed[validation] - h[validation]) ** 2).mean(axis=1), weights=validation_weights)
                denominator = np.average(((h[validation] - scaler.mean) ** 2).mean(axis=1), weights=validation_weights)
                rows.append({"preprocessing": mode, "pca_rank": rank,
                    "validation_action_clipped_mae": float(np.average(abs(error).mean(axis=1), weights=validation_weights)),
                    "validation_action_mean_mae": float(np.average(abs(new_mu[validation] - original_mu[validation]).mean(axis=1), weights=validation_weights)),
                    "validation_nmse_raw": float(mse / denominator) if denominator > 1e-15 else None})
    with (destination / "pca_fidelity.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "pca_sources.json").write_text(json.dumps({"run": str(run.resolve()), "fitted_on": "train only", "ranks": list(ranks),
            "evaluation": "validation trajectory-weighted; no state labels used to fit PCA",
            "basis_capacity_note": "rank-r PCA has r fixed directions; width-m TopK SAE selects up to K among m directions; these are different capacity constraints"}, indent=2), encoding="utf-8")
    return rows
