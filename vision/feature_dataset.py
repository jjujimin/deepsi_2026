import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class MobileNetFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_dir = Path(feature_dir)
        manifest_path = self.feature_dir / "manifest.json"

        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
            self.samples = [
                {
                    "feature_path": Path(sample["feature_path"]),
                    "label": int(sample["label"]),
                    "source_dir": sample.get("source_dir"),
                }
                for sample in manifest["samples"]
            ]
        else:
            self.samples = [
                {"feature_path": feature_path, "label": None, "source_dir": None}
                for feature_path in sorted(self.feature_dir.rglob("*.pt"))
            ]

        if not self.samples:
            raise FileNotFoundError(f"No feature files found in {self.feature_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            data = torch.load(sample["feature_path"], map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(sample["feature_path"], map_location="cpu")

        features = data["features"].float()
        label = sample["label"] if sample["label"] is not None else int(data["label"])
        return features, torch.tensor(label, dtype=torch.long)


def create_feature_dataloader(
    feature_dir,
    batch_size=32,
    workers=4,
    shuffle=True,
):
    dataset = MobileNetFeatureDataset(feature_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
        **({"prefetch_factor": 2} if workers > 0 else {}),
    )
