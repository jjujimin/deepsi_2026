import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MultiModalFeatureDataset(Dataset):
    def __init__(self, feature_dir, cache_sensor=True):
        self.feature_dir = Path(feature_dir)
        self.cache_sensor = cache_sensor
        self.sensor_cache = {}
        self.data_types = [
            "Segment Acceleration",
            "Segment Angular Velocity",
            "Sensor Magnetic Field",
        ]
        self.body_parts = [
            "Pelvis", "Head", "Left Forearm", "Left Lower Leg", "Left Shoulder",
            "Left Upper Arm", "Left Upper Leg", "Right Forearm", "Right Lower Leg",
            "Right Shoulder", "Right Upper Arm", "Right Upper Leg",
        ]

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

    def _resolve_sensor_path(self, source_dir):
        sensor_dir = Path(str(source_dir).replace("이미지", "센서"))
        sensor_files = sorted(sensor_dir.glob("*.csv"))
        if not sensor_files:
            raise FileNotFoundError(f"No sensor csv found in {sensor_dir}")
        return sensor_files[0]

    def _load_sensor_tensor(self, sensor_path):
        df = pd.read_csv(sensor_path)

        body_part_tensors = []
        for body_part in self.body_parts:
            part_cols = []
            for data_type in self.data_types:
                part_cols += [
                    col for col in df.columns
                    if data_type in col and body_part in col
                ]

            part_data = df[part_cols].values
            part_data_avg = part_data.reshape(10, 60, -1).mean(axis=1)
            body_part_tensor = torch.tensor(part_data_avg, dtype=torch.float32)
            body_part_tensors.append(body_part_tensor)

        return torch.stack(body_part_tensors)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            data = torch.load(sample["feature_path"], map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(sample["feature_path"], map_location="cpu")

        features = data["features"].float()
        label = sample["label"] if sample["label"] is not None else int(data["label"])
        source_dir = sample["source_dir"]
        if source_dir is None:
            image_paths = data.get("image_paths", [])
            source_dir = str(Path(image_paths[0]).parent) if image_paths else None
        if source_dir is None:
            raise ValueError(f"Cannot resolve sensor path for {sample['feature_path']}")

        sensor_path = self._resolve_sensor_path(source_dir)
        if self.cache_sensor:
            cache_key = str(sensor_path)
            if cache_key not in self.sensor_cache:
                self.sensor_cache[cache_key] = self._load_sensor_tensor(sensor_path)
            sensor = self.sensor_cache[cache_key]
        else:
            sensor = self._load_sensor_tensor(sensor_path)

        return features, sensor, torch.tensor(label, dtype=torch.long)


def create_feature_dataloader(
    feature_dir,
    batch_size=32,
    workers=4,
    shuffle=True,
    cache_sensor=True,
):
    dataset = MultiModalFeatureDataset(feature_dir, cache_sensor=cache_sensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
        **({"prefetch_factor": 2} if workers > 0 else {}),
    )
