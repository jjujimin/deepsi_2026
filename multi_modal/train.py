import copy
import sys
from datetime import datetime
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

from feature_dataset import create_feature_dataloader


SEED = 42
BATCH_SIZE = 32
IMAGE_SIZE = 160
NUM_WORKERS = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MODALITY_DROPOUT_PROBS = {
    "both": 0.9,
    "vision_only": 0.05,
    "sensor_only": 0.05,
}
MODEL_SAVE_PATH = "./model.pth"
LOG_DIR = Path(__file__).resolve().parent / "logs"


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


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    print(f"Logging to {log_path}")
    return log_file


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultimodalTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, video_src, sensor_src):
        output = video_src
        
        for i, layer in enumerate(self.layers):
            if i == 0:
                output = layer(output)
                sensor_output = layer(sensor_src)
            else:
                output = layer(output + sensor_output)

        return output


class MultiModalNet(nn.Module):
    def __init__(self, 
            num_classes=1, 
            d_model=512,
            nhead=8,
            num_layers=2,
            dim_feedforward=2048,
            dropout=0.1,
            pretrained_backbone=True,
            backbone_name="mobilenetv4_conv_medium",
            freeze_backbone=True,
            feature_dim=None,
            use_backbone=True,
            num_parts=12, 
            channels_per_part=9,
    ):
        super().__init__()

        if use_backbone and pretrained_backbone:
            try:
                backbone = timm.create_model(
                    backbone_name,
                    pretrained=True,
                    num_classes=0,
                    global_pool="avg",
                )
                print(f"Using ImageNet-pretrained {backbone_name} backbone")
            except Exception as exc:
                print(f"Falling back to randomly initialized {backbone_name} backbone: {exc}")
                backbone = timm.create_model(
                    backbone_name,
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                )
        elif use_backbone:
            backbone = timm.create_model(
                backbone_name,
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )

        if use_backbone:
            self.frame_encoder = backbone
            if freeze_backbone:
                for parameter in self.frame_encoder.parameters():
                    parameter.requires_grad = False

            backbone_was_training = self.frame_encoder.training
            self.frame_encoder.eval()
            with torch.no_grad():
                backbone_features = self.frame_encoder(
                    torch.zeros(1, 3, 160, 160)
                ).flatten(1).shape[1]
            self.frame_encoder.train(backbone_was_training)
        else:
            self.frame_encoder = None
            backbone_features = feature_dim
            if backbone_features is None:
                raise ValueError("feature_dim is required when use_backbone=False")

        self.video_feature_projection = (
            nn.Identity()
            if backbone_features == d_model
            else nn.Linear(backbone_features, d_model)
        )
        self.video_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.video_pos_encoder = nn.Parameter(torch.randn(1, 11, d_model))

        self.part_embedding = nn.Linear(channels_per_part, 16)
        self.full_body_embedding = nn.Linear(num_parts * 16, d_model)
        self.sensor_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.sensor_pos_encoder = nn.Parameter(torch.randn(1, 11, d_model)) # class token + 10개 타임스텝

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.multimodal_temporal_encoder = MultimodalTransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, video_x, sensor_x):
        if video_x.dim() == 5:
            batch_size, channels, frames, height, width = video_x.shape
            video_x = video_x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)

            frame_features = self.frame_encoder(video_x)
            frame_features = frame_features.flatten(1)
            frame_features = self.video_feature_projection(frame_features)
            frame_features = frame_features.view(batch_size, frames, -1)
        elif video_x.dim() == 3:
            batch_size, frames, _ = video_x.shape
            frame_features = self.video_feature_projection(video_x)
        else:
            raise ValueError(
                "Expected image input [B, C, T, H, W] or "
                f"feature input [B, T, F], got {tuple(video_x.shape)}"
            )
        video_cls_token = self.video_cls_token.expand(batch_size, -1, -1)
        temporal_features = torch.cat([video_cls_token, frame_features], dim=1)
        temporal_features = temporal_features + self.video_pos_encoder

        sensor_x = sensor_x.transpose(1, 2)
        sensor_x = self.part_embedding(sensor_x)
        sensor_x = sensor_x.reshape(batch_size, frames, -1)
        sensor_x = self.full_body_embedding(sensor_x)
        sensor_cls_token = self.sensor_cls_token.expand(batch_size, -1, -1)
        sensor_x = torch.cat([sensor_cls_token, sensor_x], dim=1)
        sensor_x = sensor_x + self.sensor_pos_encoder

        fused = self.multimodal_temporal_encoder(temporal_features, sensor_x)
        cls_feature = fused[:, 0]
        return self.fusion_head(cls_feature)


def compute_metrics(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.numel() * 100

    labels_np = labels.cpu().numpy().astype(int).ravel()
    preds_np = preds.cpu().numpy().astype(int).ravel()
    f1 = f1_score(labels_np, preds_np, zero_division=0)
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    return accuracy, f1, cm


def val(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", leave=False)
        for video, sensor, label in progress_bar:
            video = video.to(DEVICE, non_blocking=True)
            sensor = sensor.to(DEVICE, non_blocking=True)
            label = label.float().unsqueeze(1).to(DEVICE, non_blocking=True)

            output = model(video, sensor)
            loss = criterion(output, label)

            val_loss += loss.item()
            all_logits.append(output.cpu())
            all_labels.append(label.cpu())

    avg_loss = val_loss / len(val_loader)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    accuracy, f1, cm = compute_metrics(logits, labels)

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    return avg_loss, accuracy, f1


def apply_modality_dropout(video, sensor, probs):
    sample = random.random()
    if sample < probs["both"]:
        return video, sensor, "both"
    if sample < probs["both"] + probs["vision_only"]:
        return video, torch.zeros_like(sensor), "vision_only"
    return torch.zeros_like(video), sensor, "sensor_only"


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_logits = []
        all_labels = []
        modality_counts = {"both": 0, "vision_only": 0, "sensor_only": 0}
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for video, sensor, label in progress_bar:
            video = video.to(DEVICE, non_blocking=True)
            sensor = sensor.to(DEVICE, non_blocking=True)
            label = label.float().unsqueeze(1).to(DEVICE, non_blocking=True)

            video, sensor, modality_mode = apply_modality_dropout(
                video,
                sensor,
                MODALITY_DROPOUT_PROBS,
            )
            modality_counts[modality_mode] += 1

            optimizer.zero_grad(set_to_none=True)
            output = model(video, sensor)
            loss = criterion(output, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            all_logits.append(output.detach().cpu())
            all_labels.append(label.detach().cpu())
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        train_loss = running_loss / len(train_loader)
        train_logits = torch.cat(all_logits)
        train_labels = torch.cat(all_labels)
        train_acc, train_f1, _ = compute_metrics(train_logits, train_labels)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, F1 Score: {train_f1:.4f}"
        )
        print(
            "Modality Dropout "
            f"both={modality_counts['both']}, "
            f"vision_only={modality_counts['vision_only']}, "
            f"sensor_only={modality_counts['sensor_only']}"
        )

        avg_val_loss, val_acc, val_f1 = val(model, val_loader, criterion)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_val_loss,
                    "accuracy": val_acc,
                    "f1": val_f1,
                },
                MODEL_SAVE_PATH,
            )
            print(f"Best model saved to {MODEL_SAVE_PATH} (F1: {best_val_f1:.4f})")


def main():
    log_file = setup_logging()
    try:
        seed_everything(SEED)

        train_loader = create_feature_dataloader(
            "../data/cache/vision_mobilenetv4_conv_medium_160/Training",
            batch_size=BATCH_SIZE,
            workers=NUM_WORKERS,
            shuffle=True,
        )
        val_loader = create_feature_dataloader(
            "../data/cache/vision_mobilenetv4_conv_medium_160/Validation",
            batch_size=BATCH_SIZE,
            workers=NUM_WORKERS,
            shuffle=False,
        )

        model = MultiModalNet(
            num_classes=1,
            pretrained_backbone=False,
            use_backbone=False,
            feature_dim=1280,
        ).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs=NUM_EPOCHS,
        )
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
