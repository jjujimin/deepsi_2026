import torch
import torch.nn as nn
import torch.optim as optim
import timm

from tqdm import tqdm

from feature_dataset import create_feature_dataloader

import pdb

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MobileNetV4TemporalTransformer(nn.Module):
    def __init__(
        self,
        num_classes=1,
        d_model=512,
        nhead=8,
        num_transformer_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        pretrained_backbone=True,
        backbone_name="mobilenetv4_conv_medium",
        freeze_backbone=True,
        feature_dim=None,
        use_backbone=True,
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
        self.feature_projection = (
            nn.Identity()
            if backbone_features == d_model
            else nn.Linear(backbone_features, d_model)
        )
        self.video_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if x.dim() == 5:
            batch_size, channels, frames, height, width = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)

            frame_features = self.frame_encoder(x)
            frame_features = frame_features.flatten(1)
            frame_features = self.feature_projection(frame_features)
            frame_features = frame_features.view(batch_size, frames, -1)
        elif x.dim() == 3:
            frame_features = self.feature_projection(x)
        else:
            raise ValueError(
                "Expected image input [B, C, T, H, W] or "
                f"feature input [B, T, F], got {tuple(x.shape)}"
            )

        cls_token = self.video_cls_token.expand(frame_features.size(0), -1, -1)
        temporal_features = torch.cat([cls_token, frame_features], dim=1)
        temporal_features = self.positional_encoding(temporal_features)
        temporal_features = self.temporal_encoder(temporal_features)
        video_features = temporal_features[:, 0]

        return self.classifier(video_features)



ResNet18TemporalTransformer = MobileNetV4TemporalTransformer


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


criterion = torch.nn.BCEWithLogitsLoss()

def val(model, val_loader, criterion, device='cpu'):
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    with torch.no_grad(): 
        for img, labels in tqdm(val_loader):
            img = img.to(device)
            labels = labels.to(device, dtype=torch.float32)

            output = model(img)
            loss = criterion(output.view(-1), labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(output.view(-1)) > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
            true_positive += ((preds == 1) & (labels == 1)).sum().item()
            false_positive += ((preds == 1) & (labels == 0)).sum().item()
            true_negative += ((preds == 0) & (labels == 0)).sum().item()
            false_negative += ((preds == 0) & (labels == 1)).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_preds / total_preds * 100
    precision = true_positive / max(true_positive + false_positive, 1) * 100
    recall = true_positive / max(true_positive + false_negative, 1) * 100
    specificity = true_negative / max(true_negative + false_positive, 1) * 100
    f1 = (
        2 * precision * recall / max(precision + recall, 1e-8)
        if precision + recall > 0
        else 0.0
    )

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    print(
        "Validation "
        f"Precision: {precision:.2f}%, Recall: {recall:.2f}%, "
        f"Specificity: {specificity:.2f}%, F1: {f1:.2f}%"
    )
    print(
        "Confusion Matrix "
        f"TN={true_negative}, FP={false_positive}, FN={false_negative}, TP={true_positive}"
    )
    
    return avg_val_loss, val_accuracy

def train(model, train_loader, val_loader, optimizer, criterion, device='cpu', epoches=10):
    best_val_acc = 0
    for epoch in range(epoches):
        model.train()
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for img, labels in tqdm(train_loader):
            img = img.to(device)
            labels = labels.to(device, dtype=torch.float32) 
            optimizer.zero_grad()
            
            output = model(img)
            
            loss = criterion(output.view(-1), labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            preds = (torch.sigmoid(output.view(-1)) > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct_preds / total_preds * 100

        print(f"Epoch [{epoch+1}/{epoches}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        avg_val_loss, val_acc = val(model, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = 'model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }, model_save_path)
            print(f"Model and parameters saved to {model_save_path}")
            
    return avg_val_loss, val_acc
        
def main():
    device = get_device()
    print(f"Using {device} device")
    train_feature_dir = "../data/cache/vision_mobilenetv4_conv_medium_160/Training"
    val_feature_dir = "../data/cache/vision_mobilenetv4_conv_medium_160/Validation"

    model = MobileNetV4TemporalTransformer(
        num_classes=1,
        pretrained_backbone=False,
        use_backbone=False,
        feature_dim=1280,
    ).to(device)

    train_dataloader = create_feature_dataloader(
        train_feature_dir,
        batch_size=32,
        workers=4,
        shuffle=True,
    )
    val_dataloader = create_feature_dataloader(
        val_feature_dir,
        batch_size=32,
        workers=4,
        shuffle=False,
    )

    learning_rate = 0.0001
    epochs = 10
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        device=device,
        epoches=epochs,
    )


if __name__ == "__main__":
    main()
