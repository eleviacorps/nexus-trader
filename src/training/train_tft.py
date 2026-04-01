from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

import numpy as np

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # type: ignore
except ImportError:  # pragma: no cover
    accuracy_score = None
    f1_score = None
    roc_auc_score = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    patience: int = 3
    batch_size: int = 512
    inherited_lr: float = 1e-4
    new_layers_lr: float = 5e-4


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def sim_weight_from_confidence(confidence: float) -> float:
    return clamp((confidence - 0.3) / 0.7, 0.0, 1.0)


def combined_loss_weights(confidence: float) -> Dict[str, float]:
    sim_weight = sim_weight_from_confidence(confidence)
    denominator = 3.0 + sim_weight
    return {
        "real_weight": 3.0 / denominator,
        "sim_weight": sim_weight / denominator,
    }


def save_feature_importance_report(path: Path, report: Mapping[str, float]) -> None:
    path.write_text(json.dumps(dict(report), indent=2), encoding="utf-8")


def save_training_config(path: Path, config: TrainingConfig) -> None:
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def save_json_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def build_optimizer(model: Any, old_layers_lr: float = 1e-4, new_layers_lr: float = 5e-4):
    if torch is None:
        raise ImportError("PyTorch is required to build the optimizer.")
    if hasattr(model, "optimizer_groups"):
        return torch.optim.AdamW(model.optimizer_groups(old_layers_lr, new_layers_lr))
    return torch.optim.AdamW(model.parameters(), lr=old_layers_lr)


def weighted_binary_loss(predictions, targets, sim_targets=None, sim_confidence=None, sample_weights=None):
    if torch is None:
        raise ImportError("PyTorch is required for loss computation.")
    if sample_weights is None:
        real_loss = F.binary_cross_entropy(predictions, targets)
    else:
        real_loss = F.binary_cross_entropy(predictions, targets, reduction="none")
        real_loss = (real_loss * sample_weights).mean()
    if sim_targets is None or sim_confidence is None:
        return real_loss
    sim_weight = torch.clamp((sim_confidence - 0.3) / 0.7, 0.0, 1.0)
    sim_loss = F.binary_cross_entropy(predictions, sim_targets, reduction="none")
    if sample_weights is not None:
        sim_loss = sim_loss * sample_weights
    combined = (3.0 * real_loss + (sim_weight * sim_loss).mean()) / (3.0 + sim_weight.mean().clamp(min=1e-6))
    return combined


def weighted_multihorizon_loss(
    predictions,
    targets,
    sim_targets=None,
    sim_confidence=None,
    sample_weights=None,
):
    if torch is None:
        raise ImportError("PyTorch is required for loss computation.")
    real_loss = F.binary_cross_entropy(predictions, targets, reduction="none")
    real_loss = real_loss.mean(dim=1)
    if sample_weights is not None:
        real_loss = real_loss * sample_weights
    real_loss = real_loss.mean()
    if sim_targets is None or sim_confidence is None:
        return real_loss
    primary_predictions = predictions[:, 0]
    sim_weight = torch.clamp((sim_confidence - 0.3) / 0.7, 0.0, 1.0)
    sim_loss = F.binary_cross_entropy(primary_predictions, sim_targets, reduction="none")
    if sample_weights is not None:
        sim_loss = sim_loss * sample_weights
    combined = (3.0 * real_loss + (sim_weight * sim_loss).mean()) / (3.0 + sim_weight.mean().clamp(min=1e-6))
    return combined


def collect_binary_metrics(targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = (probabilities >= threshold).astype(np.float32)
    metrics: Dict[str, float] = {
        "positive_rate": float(targets.mean()) if targets.size else 0.0,
        "brier_score": float(np.mean((probabilities - targets) ** 2)) if targets.size else 0.0,
        "threshold": float(threshold),
    }

    if accuracy_score is not None:
        metrics["accuracy"] = float(accuracy_score(targets, labels))
    else:
        metrics["accuracy"] = float((labels == targets).mean()) if targets.size else 0.0

    if f1_score is not None:
        metrics["f1"] = float(f1_score(targets, labels, zero_division=0))
    else:
        tp = float(((labels == 1) & (targets == 1)).sum())
        fp = float(((labels == 1) & (targets == 0)).sum())
        fn = float(((labels == 0) & (targets == 1)).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        metrics["f1"] = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    if roc_auc_score is not None and len(np.unique(targets)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def find_optimal_threshold(targets: np.ndarray, probabilities: np.ndarray, metric: str = "accuracy") -> Dict[str, float]:
    targets = np.asarray(targets, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if targets.size == 0 or probabilities.size == 0:
        return {"threshold": 0.5, "score": 0.0, "metric": metric}

    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in np.linspace(0.05, 0.95, 181):
        threshold = float(threshold)
        metrics = collect_binary_metrics(targets, probabilities, threshold=threshold)
        score = float(metrics.get(metric, 0.0))
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return {"threshold": best_threshold, "score": best_score, "metric": metric}


def build_calibration_report(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> Dict[str, Any]:
    targets = np.asarray(targets, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if targets.size == 0 or probabilities.size == 0:
        return {"bins": [], "ece": 0.0, "max_calibration_gap": 0.0}

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    max_gap = 0.0
    total = max(1, len(targets))
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)
        count = int(mask.sum())
        if count == 0:
            continue
        bucket_targets = targets[mask]
        bucket_probabilities = probabilities[mask]
        observed = float(bucket_targets.mean())
        predicted = float(bucket_probabilities.mean())
        gap = abs(observed - predicted)
        ece += gap * (count / total)
        max_gap = max(max_gap, gap)
        rows.append(
            {
                "left": float(left),
                "right": float(right),
                "count": count,
                "predicted_mean": predicted,
                "observed_rate": observed,
                "gap": float(gap),
            }
        )
    return {"bins": rows, "ece": float(ece), "max_calibration_gap": float(max_gap)}


def collect_multihorizon_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Sequence[float] | None = None,
    horizon_labels: Sequence[str] | None = None,
) -> Dict[str, Any]:
    targets = np.asarray(targets, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if targets.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("Multi-horizon metrics require 2D target/probability arrays.")
    if targets.shape != probabilities.shape:
        raise ValueError("Multi-horizon targets/probabilities must share shape.")
    output_dim = targets.shape[1]
    threshold_values = list(thresholds or [0.5] * output_dim)
    if len(threshold_values) != output_dim:
        raise ValueError("threshold count must match multi-horizon output dimension")
    labels = list(horizon_labels or [f"h{i}" for i in range(output_dim)])
    horizon_metrics: Dict[str, Dict[str, float]] = {}
    primary_metrics = collect_binary_metrics(targets[:, 0], probabilities[:, 0], threshold=threshold_values[0])
    for index, label in enumerate(labels):
        horizon_metrics[str(label)] = collect_binary_metrics(targets[:, index], probabilities[:, index], threshold=threshold_values[index])
    return {
        "primary": primary_metrics,
        "horizons": horizon_metrics,
    }


def horizon_agreement_features(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = (probabilities >= threshold).astype(np.float32)
    primary = labels[:, :1]
    agreement = (labels == primary).mean(axis=1, keepdims=True)
    centered = np.abs(probabilities - 0.5)
    return np.concatenate(
        [
            probabilities[:, :1],
            probabilities.mean(axis=1, keepdims=True),
            probabilities.std(axis=1, keepdims=True),
            centered.max(axis=1, keepdims=True),
            centered.mean(axis=1, keepdims=True),
            agreement,
        ],
        axis=1,
    ).astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def train_precision_gate(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float = 0.5,
    epochs: int = 300,
    lr: float = 0.05,
    l2: float = 1e-3,
) -> Dict[str, Any]:
    features = horizon_agreement_features(probabilities, threshold=threshold)
    labels = ((probabilities[:, 0] >= threshold).astype(np.float32) == targets[:, 0].astype(np.float32)).astype(np.float32)
    if len(np.unique(labels)) < 2:
        return {
            "feature_names": ["primary_prob", "mean_prob", "std_prob", "max_confidence", "mean_confidence", "agreement_ratio"],
            "weights": [0.0] * features.shape[1],
            "bias": float(np.log(np.clip(labels.mean() / max(1e-6, 1.0 - labels.mean()), 1e-6, 1e6))) if labels.size else 0.0,
            "train_accuracy": float(labels.mean()) if labels.size else 0.0,
            "threshold": 0.5,
        }
    weights = np.zeros(features.shape[1], dtype=np.float32)
    bias = 0.0
    for _ in range(epochs):
        logits = features @ weights + bias
        preds = _sigmoid(logits)
        error = preds - labels
        grad_w = (features.T @ error) / max(1, len(features)) + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    final_probs = _sigmoid(features @ weights + bias)
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold_candidate in np.linspace(0.5, 0.9, 17):
        active = final_probs >= float(threshold_candidate)
        participation = float(active.mean()) if active.size else 0.0
        if participation < 0.05:
            continue
        precision = float(labels[active].mean()) if active.any() else 0.0
        score = (0.75 * precision) + (0.25 * participation)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold_candidate)
    return {
        "feature_names": ["primary_prob", "mean_prob", "std_prob", "max_confidence", "mean_confidence", "agreement_ratio"],
        "weights": [float(value) for value in weights],
        "bias": float(bias),
        "train_accuracy": float((((final_probs >= 0.5).astype(np.float32)) == labels).mean()),
        "threshold": float(best_threshold),
        "train_participation": float((final_probs >= best_threshold).mean()) if final_probs.size else 0.0,
        "train_precision": float(labels[final_probs >= best_threshold].mean()) if np.any(final_probs >= best_threshold) else 0.0,
    }


def apply_precision_gate(probabilities: np.ndarray, gate: Mapping[str, Any]) -> np.ndarray:
    features = horizon_agreement_features(probabilities)
    weights = np.asarray(gate.get("weights", []), dtype=np.float32)
    bias = float(gate.get("bias", 0.0))
    if weights.size != features.shape[1]:
        raise ValueError("Precision gate weight dimension does not match expected feature count.")
    return _sigmoid(features @ weights + bias).astype(np.float32)


def evaluate_binary_model(model, dataloader, device, threshold: float = 0.5) -> tuple[Dict[str, float], np.ndarray, np.ndarray]:
    if torch is None:
        raise ImportError("PyTorch is required for evaluation.")

    model.eval()
    all_targets = []
    all_probs = []
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            sample_weights = None
            if len(batch) == 2:
                features, targets = batch
                sim_targets = sim_conf = None
            elif len(batch) == 3:
                features, targets, sample_weights = batch
                sim_targets = sim_conf = None
            elif len(batch) == 4:
                features, targets, sim_targets, sim_conf = batch
            else:
                features, targets, sim_targets, sim_conf, sample_weights = batch
            features = features.to(device)
            targets = targets.to(device)
            probs = model(features)
            loss = weighted_binary_loss(
                probs,
                targets,
                sim_targets.to(device) if sim_targets is not None else None,
                sim_conf.to(device) if sim_conf is not None else None,
                sample_weights.to(device) if sample_weights is not None else None,
            )
            losses.append(float(loss.item()))
            all_targets.append(targets.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
    targets_np = np.concatenate(all_targets) if all_targets else np.empty(0, dtype=np.float32)
    probs_np = np.concatenate(all_probs) if all_probs else np.empty(0, dtype=np.float32)
    metrics = collect_binary_metrics(targets_np, probs_np, threshold=threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, targets_np, probs_np


def evaluate_multihorizon_model(model, dataloader, device, thresholds: Sequence[float] | None = None, horizon_labels: Sequence[str] | None = None):
    if torch is None:
        raise ImportError("PyTorch is required for evaluation.")
    all_targets = []
    all_probs = []
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            sample_weights = None
            if len(batch) == 2:
                features, targets = batch
                sim_targets = sim_conf = None
            elif len(batch) == 3:
                features, targets, sample_weights = batch
                sim_targets = sim_conf = None
            elif len(batch) == 4:
                features, targets, sim_targets, sim_conf = batch
            else:
                features, targets, sim_targets, sim_conf, sample_weights = batch
            features = features.to(device)
            targets = targets.to(device)
            probs = model(features)
            loss = weighted_multihorizon_loss(
                probs,
                targets,
                sim_targets.to(device) if sim_targets is not None else None,
                sim_conf.to(device) if sim_conf is not None else None,
                sample_weights.to(device) if sample_weights is not None else None,
            )
            losses.append(float(loss.item()))
            all_targets.append(targets.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
    targets_np = np.concatenate(all_targets) if all_targets else np.empty((0, 0), dtype=np.float32)
    probs_np = np.concatenate(all_probs) if all_probs else np.empty((0, 0), dtype=np.float32)
    metrics = collect_multihorizon_metrics(targets_np, probs_np, thresholds=thresholds, horizon_labels=horizon_labels)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, targets_np, probs_np


def train_binary_model(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    epochs: int,
    patience: int,
    selection_metric: str = "accuracy",
):
    if torch is None:
        raise ImportError("PyTorch is required for training.")

    best_state = None
    best_metrics: Dict[str, float] | None = None
    best_score = float("-inf")
    patience_left = patience
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            sample_weights = None
            if len(batch) == 2:
                features, targets = batch
                sim_targets = sim_conf = None
            elif len(batch) == 3:
                features, targets, sample_weights = batch
                sim_targets = sim_conf = None
            elif len(batch) == 4:
                features, targets, sim_targets, sim_conf = batch
            else:
                features, targets, sim_targets, sim_conf, sample_weights = batch
            features = features.to(device)
            targets = targets.to(device)
            sim_targets = sim_targets.to(device) if sim_targets is not None else None
            sim_conf = sim_conf.to(device) if sim_conf is not None else None
            sample_weights = sample_weights.to(device) if sample_weights is not None else None

            optimizer.zero_grad(set_to_none=True)
            probs = model(features)
            loss = weighted_binary_loss(probs, targets, sim_targets, sim_conf, sample_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics, _, _ = evaluate_binary_model(model, val_loader, device)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(epoch_record)

        score = val_metrics.get(selection_metric, 0.0)
        if score > best_score:
            best_score = score
            patience_left = patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = val_metrics
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, (best_metrics or {})


def train_multihorizon_model(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    epochs: int,
    patience: int,
    selection_metric: str = "accuracy",
    horizon_labels: Sequence[str] | None = None,
):
    if torch is None:
        raise ImportError("PyTorch is required for training.")

    best_state = None
    best_metrics: Dict[str, Any] | None = None
    best_score = float("-inf")
    patience_left = patience
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            sample_weights = None
            if len(batch) == 2:
                features, targets = batch
                sim_targets = sim_conf = None
            elif len(batch) == 3:
                features, targets, sample_weights = batch
                sim_targets = sim_conf = None
            elif len(batch) == 4:
                features, targets, sim_targets, sim_conf = batch
            else:
                features, targets, sim_targets, sim_conf, sample_weights = batch
            features = features.to(device)
            targets = targets.to(device)
            sim_targets = sim_targets.to(device) if sim_targets is not None else None
            sim_conf = sim_conf.to(device) if sim_conf is not None else None
            sample_weights = sample_weights.to(device) if sample_weights is not None else None

            optimizer.zero_grad(set_to_none=True)
            probs = model(features)
            loss = weighted_multihorizon_loss(probs, targets, sim_targets, sim_conf, sample_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        val_metrics, _, _ = evaluate_multihorizon_model(model, val_loader, device, horizon_labels=horizon_labels)
        primary_metrics = val_metrics.get("primary", {})
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_record = {"epoch": epoch, "train_loss": train_loss, **primary_metrics}
        history.append(epoch_record)

        score = float(primary_metrics.get(selection_metric, 0.0))
        if score > best_score:
            best_score = score
            patience_left = patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = val_metrics
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, (best_metrics or {})
