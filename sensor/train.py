import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from sensordataset import SensorDataset


data_types = ['Segment Acceleration', 'Segment Angular Velocity', 'Sensor Magnetic Field']
train_dataset = SensorDataset(input_dir='../data/Training/01.원천데이터/센서', data_types=data_types)
val_dataset = SensorDataset(input_dir='../data/Validation/01.원천데이터/센서', data_types=data_types)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers = 4)


class SpatialTemporalFallTransformer(nn.Module):
    def __init__(self, num_parts=12, channels_per_part=9, d_model=128, nhead=8, num_layers=2):
        super(SpatialTemporalFallTransformer, self).__init__()

        self.part_embedding = nn.Linear(channels_per_part, 16)

        self.full_body_embedding = nn.Linear(num_parts * 16, d_model)

        self.pos_encoder = nn.Parameter(torch.randn(1, 10, d_model))

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
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
        out = self.classifier(x)

        return out
    
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = SpatialTemporalFallTransformer().to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

num_epochs = 30 
best_val_f1 = 0.0
early_stop_patience = 7
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for samples, labels in train_loader:
        samples, labels = samples.to(device), labels.float().to(device).unsqueeze(1)

        logits = model(samples)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * samples.size(0) 

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_train / total_train
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

    model.eval() 
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    all_val_labels = []
    all_val_predictions = []

    with torch.no_grad(): 
        for val_samples, val_labels in val_loader:
            val_samples, val_labels = val_samples.to(device), val_labels.float().to(device).unsqueeze(1)

            val_logits = model(val_samples)
            val_loss += criterion(val_logits, val_labels).item() * val_samples.size(0)

            val_probabilities = torch.sigmoid(val_logits)
            val_predictions = (val_probabilities >= 0.5).float()
            correct_val += (val_predictions == val_labels).sum().item()
            total_val += val_labels.size(0)

            all_val_labels.extend(val_labels.cpu().numpy().ravel())
            all_val_predictions.extend(val_predictions.cpu().numpy().ravel())

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct_val / total_val
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    cm = confusion_matrix(all_val_labels, all_val_predictions)
    precision = precision_score(all_val_labels, all_val_predictions, zero_division=0)
    recall = recall_score(all_val_labels, all_val_predictions, zero_division=0)
    f1 = f1_score(all_val_labels, all_val_predictions, zero_division=0)
    scheduler.step(f1)

    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    if f1 > best_val_f1:
        best_val_f1 = f1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), './best_model.pth')
        print("Best model updated and saved!")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}. Best Validation F1: {best_val_f1:.4f}")
            break

print("Training complete.")
