import argparse
import importlib.util
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = REPO_ROOT / "vision"
MULTI_MODAL_DIR = REPO_ROOT / "multi_modal"


def load_module(module_name, module_path):
    module_path = Path(module_path)
    module_dir = str(module_path.parent)
    old_path = list(sys.path)
    sys.path.insert(0, module_dir)
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = old_path


vision_train = load_module("vision_train_for_distill", VISION_DIR / "train.py")
multi_train = load_module("multi_modal_train_for_distill", MULTI_MODAL_DIR / "train.py")
multi_features = load_module("multi_modal_features_for_distill", MULTI_MODAL_DIR / "feature_dataset.py")

MobileNetV4TemporalTransformer = vision_train.MobileNetV4TemporalTransformer
MultiModalNet = multi_train.MultiModalNet
create_feature_dataloader = multi_features.create_feature_dataloader


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = log_path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.__stdout__, log_file)
    print(f"Logging to {log_path}")
    return log_file, original_stdout


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(device_name):
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

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def build_teacher(args, device):
    teacher = MultiModalNet(
        num_classes=1,
        pretrained_backbone=False,
        use_backbone=False,
        feature_dim=args.feature_dim,
    ).to(device)
    teacher.load_state_dict(load_state_dict(args.teacher_checkpoint, device))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def build_student(args, device):
    student = MobileNetV4TemporalTransformer(
        num_classes=1,
        pretrained_backbone=False,
        use_backbone=False,
        feature_dim=args.feature_dim,
    ).to(device)
    if args.student_init_checkpoint is not None:
        student.load_state_dict(load_state_dict(args.student_init_checkpoint, device), strict=False)
    return student


def get_student_transformer_tokens(student, video_features):
    frame_features = student.feature_projection(video_features)
    cls_token = student.video_cls_token.expand(frame_features.size(0), -1, -1)
    tokens = torch.cat([cls_token, frame_features], dim=1)
    tokens = student.positional_encoding(tokens)
    return student.temporal_encoder(tokens)


def get_teacher_transformer_tokens(teacher, video_features, sensor):
    batch_size, frames, _ = video_features.shape

    frame_features = teacher.video_feature_projection(video_features)
    video_cls_token = teacher.video_cls_token.expand(batch_size, -1, -1)
    video_tokens = torch.cat([video_cls_token, frame_features], dim=1)
    video_tokens = video_tokens + teacher.video_pos_encoder

    sensor_tokens = sensor.transpose(1, 2)
    sensor_tokens = teacher.part_embedding(sensor_tokens)
    sensor_tokens = sensor_tokens.reshape(batch_size, frames, -1)
    sensor_tokens = teacher.full_body_embedding(sensor_tokens)
    sensor_cls_token = teacher.sensor_cls_token.expand(batch_size, -1, -1)
    sensor_tokens = torch.cat([sensor_cls_token, sensor_tokens], dim=1)
    sensor_tokens = sensor_tokens + teacher.sensor_pos_encoder

    fused_tokens = teacher.multimodal_temporal_encoder(video_tokens, sensor_tokens)
    return fused_tokens[:, 1:].detach()


def compute_distillation_losses(
    student_logits,
    teacher_logits,
    student_tokens,
    teacher_tokens,
    labels,
    hard_weight,
    soft_weight,
    hidden_weight,
    temperature,
):
    hard_loss = nn.functional.binary_cross_entropy_with_logits(student_logits, labels)
    teacher_probs = torch.sigmoid(teacher_logits / temperature)
    soft_loss = nn.functional.binary_cross_entropy_with_logits(
        student_logits / temperature,
        teacher_probs,
    ) * (temperature ** 2)
    hidden_loss = nn.functional.mse_loss(student_tokens[:, 1:], teacher_tokens)
    total_loss = (
        hard_weight * hard_loss
        + soft_weight * soft_loss
        + hidden_weight * hidden_loss
    )
    return total_loss, hard_loss, soft_loss, hidden_loss


def compute_metrics(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    accuracy = (preds == labels).sum().item() / labels.numel() * 100
    labels_np = labels.cpu().numpy().astype(int).ravel()
    preds_np = preds.cpu().numpy().astype(int).ravel()
    f1 = f1_score(labels_np, preds_np, zero_division=0)
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    return accuracy, f1, cm


def validate(student, val_loader, criterion, device, threshold):
    student.eval()
    running_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for video_features, _sensor, labels in tqdm(val_loader, desc="Validating", leave=False):
            video_features = video_features.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            logits = student(video_features)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    accuracy, f1, cm = compute_metrics(logits, labels, threshold=threshold)
    avg_loss = running_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    return avg_loss, accuracy, f1


def train(args):
    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"Using {device} device")
    print(f"Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"Train features: {args.train_feature_dir}")
    print(f"Validation features: {args.val_feature_dir}")

    train_loader = create_feature_dataloader(
        args.train_feature_dir,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=True,
    )
    val_loader = create_feature_dataloader(
        args.val_feature_dir,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=False,
    )

    teacher = build_teacher(args, device)
    student = build_student(args, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(args.epochs):
        student.train()
        running_loss = 0.0
        running_hard = 0.0
        running_soft = 0.0
        running_hidden = 0.0
        all_logits = []
        all_labels = []

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.epochs}]", leave=False)
        for video_features, sensor, labels in progress_bar:
            video_features = video_features.to(device, non_blocking=True)
            sensor = sensor.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = teacher(video_features, sensor)
                teacher_tokens = get_teacher_transformer_tokens(teacher, video_features, sensor)

            student_tokens = get_student_transformer_tokens(student, video_features)
            student_logits = student.classifier(student_tokens[:, 0])
            loss, hard_loss, soft_loss, hidden_loss = compute_distillation_losses(
                student_logits,
                teacher_logits,
                student_tokens,
                teacher_tokens,
                labels,
                hard_weight=args.hard_weight,
                soft_weight=args.soft_weight,
                hidden_weight=args.hidden_weight,
                temperature=args.temperature,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()

            running_loss += loss.item()
            running_hard += hard_loss.item()
            running_soft += soft_loss.item()
            running_hidden += hidden_loss.item()
            all_logits.append(student_logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        train_logits = torch.cat(all_logits)
        train_labels = torch.cat(all_labels)
        train_acc, train_f1, _ = compute_metrics(train_logits, train_labels, threshold=args.threshold)
        num_batches = len(train_loader)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}], "
            f"Loss: {running_loss / num_batches:.4f}, "
            f"Hard: {running_hard / num_batches:.4f}, "
            f"Soft: {running_soft / num_batches:.4f}, "
            f"Hidden: {running_hidden / num_batches:.4f}, "
            f"Accuracy: {train_acc:.2f}%, F1 Score: {train_f1:.4f}"
        )

        val_loss, val_acc, val_f1 = validate(student, val_loader, criterion, device, args.threshold)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = args.output_dir / "student_vision_distilled.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": val_loss,
                    "accuracy": val_acc,
                    "f1": val_f1,
                    "teacher_checkpoint": str(args.teacher_checkpoint),
                    "hard_weight": args.hard_weight,
                    "soft_weight": args.soft_weight,
                    "hidden_weight": args.hidden_weight,
                    "temperature": args.temperature,
                    "feature_dim": args.feature_dim,
                },
                save_path,
            )
            print(f"Best student saved to {save_path} (F1: {best_f1:.4f})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill a multi-modal teacher into a vision-only student."
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=MULTI_MODAL_DIR / "model.pth",
    )
    parser.add_argument(
        "--student-init-checkpoint",
        type=Path,
        default=None,
        help="Optional vision checkpoint to initialize the student.",
    )
    parser.add_argument(
        "--train-feature-dir",
        type=Path,
        default=REPO_ROOT / "data" / "cache" / "vision_mobilenetv4_conv_medium_160" / "Training",
    )
    parser.add_argument(
        "--val-feature-dir",
        type=Path,
        default=REPO_ROOT / "data" / "cache" / "vision_mobilenetv4_conv_medium_160" / "Validation",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs")
    parser.add_argument("--feature-dim", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--soft-weight", type=float, default=0.005)
    parser.add_argument("--hidden-weight", type=float, default=0.0005)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_file, original_stdout = setup_logging(args.log_dir)
    try:
        train(args)
    finally:
        sys.stdout = original_stdout
        log_file.close()
