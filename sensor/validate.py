import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from sensordataset import SensorDataset


DATA_TYPES = [
    "Segment Acceleration",
    "Segment Angular Velocity",
    "Sensor Magnetic Field",
]


class SpatialTemporalFallTransformer(nn.Module):
    def __init__(
        self,
        num_parts=12,
        channels_per_part=9,
        d_model=128,
        nhead=8,
        num_layers=2,
    ):
        super(SpatialTemporalFallTransformer, self).__init__()

        self.part_embedding = nn.Linear(channels_per_part, 16)
        self.full_body_embedding = nn.Linear(num_parts * 16, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 10, d_model))

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = x.transpose(1, 2)
        x = self.part_embedding(x)
        x = x.reshape(batch_size, 10, -1)
        x = self.full_body_embedding(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.classifier(x)


def get_device(device_name="auto"):
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_state_dict(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    return checkpoint


def get_sample_paths(data_loader):
    dataset = data_loader.dataset

    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset
        sample_indices = dataset.indices
    else:
        base_dataset = dataset
        sample_indices = range(len(dataset))

    sample_paths = []
    for index in sample_indices:
        csv_file = base_dataset.csv_files[int(index)]
        sample_paths.append(Path(csv_file).parent if csv_file else None)

    return sample_paths


def get_fall_direction(sample_path):
    if sample_path is None:
        return "알수없음"

    direction_code = sample_path.parent.name
    direction_names = {
        "FY": "전방",
        "BY": "후방",
        "SY": "측방",
    }

    if direction_code in direction_names:
        return direction_names[direction_code]

    sample_name = sample_path.name
    for code, name in direction_names.items():
        if f"_{code}_" in sample_name:
            return name

    return "알수없음"


def compute_direction_stats(data_loader, labels, predictions):
    sample_paths = get_sample_paths(data_loader)
    stats = {
        "전방": {"total": 0, "correct": 0},
        "후방": {"total": 0, "correct": 0},
        "측방": {"total": 0, "correct": 0},
        "알수없음": {"total": 0, "correct": 0},
    }

    for sample_path, label, prediction in zip(sample_paths, labels, predictions):
        if label != 1:
            continue

        direction = get_fall_direction(sample_path)
        stats.setdefault(direction, {"total": 0, "correct": 0})
        stats[direction]["total"] += 1
        stats[direction]["correct"] += int(prediction == label)

    if stats["알수없음"]["total"] == 0:
        del stats["알수없음"]

    return stats


def validate(model, data_loader, criterion, device, threshold):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for samples, labels in data_loader:
            samples = samples.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            logits = model(samples)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).float()

            total_loss += loss.item() * samples.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().ravel())
            all_predictions.extend(predictions.cpu().numpy().ravel())
            all_probabilities.extend(probabilities.cpu().numpy().ravel())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / total
    labels_np = np.asarray(all_labels, dtype=np.int64)
    predictions_np = np.asarray(all_predictions, dtype=np.int64)
    class_labels = [1, 0]
    class_names = ["낙상", "비낙상"]
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        labels_np,
        predictions_np,
        labels=class_labels,
        zero_division=0,
    )
    per_class_metrics = {
        class_name: {
            "label": class_label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }
        for class_name, class_label, precision, recall, f1, support in zip(
            class_names,
            class_labels,
            class_precision,
            class_recall,
            class_f1,
            class_support,
        )
    }

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(labels_np, predictions_np, labels=[0, 1]),
        "precision": precision_score(labels_np, predictions_np, zero_division=0),
        "recall": recall_score(labels_np, predictions_np, zero_division=0),
        "f1": f1_score(labels_np, predictions_np, zero_division=0),
        "num_samples": total,
        "num_fall": int(labels_np.sum()),
        "num_normal": int((labels_np == 0).sum()),
        "probabilities": np.asarray(all_probabilities),
        "labels": labels_np,
        "predictions": predictions_np,
        "per_class": per_class_metrics,
        "direction_stats": compute_direction_stats(data_loader, labels_np, predictions_np),
    }
    return metrics


def print_performance_table(metrics):
    print("Sensor Model Performance")
    print(f"{'class':<8}{'precision':>12}{'recall':>12}{'F1-score':>12}{'accuracy':>12}")
    for index, class_name in enumerate(["낙상", "비낙상"]):
        class_metrics = metrics["per_class"][class_name]
        accuracy = f"{metrics['accuracy']:.4f}" if index == 0 else ""
        print(
            f"{class_name:<8}"
            f"{class_metrics['precision']:>12.4f}"
            f"{class_metrics['recall']:>12.4f}"
            f"{class_metrics['f1']:>12.4f}"
            f"{accuracy:>12}"
        )


def print_direction_table(metrics):
    print("Fall Direction Counts")
    print(f"{'direction':<8}{'total':>10}{'correct':>10}{'accuracy':>12}")
    for direction in ["전방", "후방", "측방", "알수없음"]:
        if direction not in metrics["direction_stats"]:
            continue

        stats = metrics["direction_stats"][direction]
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total else 0.0
        print(f"{direction:<8}{total:>10}{correct:>10}{accuracy:>12.4f}")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Validate sensor fall detection model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data" / "Validation" / "01.원천데이터" / "센서",
        help="Validation sensor data directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=script_dir / "best_model.pth",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Optional .csv path to save labels, predictions, and probabilities.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    dataset = SensorDataset(input_dir=str(args.data_dir), data_types=DATA_TYPES)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    try:
        model = SpatialTemporalFallTransformer().to(device)
    except torch.AcceleratorError as exc:
        if device.type != "cpu":
            print(f"Falling back to CPU because {device} initialization failed: {exc}")
            device = torch.device("cpu")
            model = SpatialTemporalFallTransformer().to(device)
        else:
            raise

    state_dict = load_state_dict(args.checkpoint, device)
    model.load_state_dict(state_dict)

    criterion = nn.BCEWithLogitsLoss()
    metrics = validate(model, data_loader, criterion, device, args.threshold)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation data: {args.data_dir}")
    print(f"Samples: {metrics['num_samples']} (Normal: {metrics['num_normal']}, Fall: {metrics['num_fall']})")
    print(f"Validation Loss: {metrics['loss']:.4f}")
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, "
        f"F1-Score: {metrics['f1']:.4f}"
    )
    print_performance_table(metrics)
    print_direction_table(metrics)

    if args.save_predictions:
        import pandas as pd

        output = args.save_predictions
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "label": metrics["labels"],
                "prediction": metrics["predictions"],
                "probability": metrics["probabilities"],
            }
        ).to_csv(output, index=False)
        print(f"Predictions saved to: {output}")


if __name__ == "__main__":
    main()
