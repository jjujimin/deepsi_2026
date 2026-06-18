import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset_for_fact_CNN import video_dataset
from train import MobileNetV4TemporalTransformer, get_device


def build_output_path(output_dir, image_root, first_frame_path):
    video_dir = Path(first_frame_path).parent
    relative_video_dir = video_dir.relative_to(image_root)
    return output_dir / relative_video_dir.with_suffix(".pt")


def collate_video_batch(batch):
    videos, labels, sample_indices = zip(*batch)
    return torch.stack(videos), torch.tensor(labels), torch.tensor(sample_indices)


class IndexedVideoDataset(video_dataset):
    def __getitem__(self, idx):
        video, label = super().__getitem__(idx)
        return video, label, idx


def save_features(args):
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = IndexedVideoDataset(
        dir=str(image_root),
        transform=transform,
        num_frames=args.num_frames,
        image_size=args.image_size,
        normalize=True,
        use_fast_decode=args.use_fast_decode,
    )
    if args.max_samples is not None:
        dataset.inputs = dataset.inputs[: args.max_samples]
        dataset.labels = dataset.labels[: args.max_samples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=args.workers > 0,
        **({"prefetch_factor": 2} if args.workers > 0 else {}),
        collate_fn=collate_video_batch,
    )

    device = args.device or get_device()
    model = MobileNetV4TemporalTransformer(
        pretrained_backbone=not args.no_pretrained,
        backbone_name=args.backbone_name,
        freeze_backbone=True,
    ).to(device)
    model.eval()
    frame_encoder = model.frame_encoder

    manifest = []
    saved_count = 0
    skipped_count = 0

    with torch.no_grad():
        for videos, labels, sample_indices in tqdm(loader, desc="Precomputing features"):
            videos = videos.to(device)
            batch_size, channels, frames, height, width = videos.shape
            flat_frames = videos.permute(0, 2, 1, 3, 4).reshape(
                batch_size * frames,
                channels,
                height,
                width,
            )

            features = frame_encoder(flat_frames).flatten(1)
            features = features.view(batch_size, frames, -1).cpu()

            for feature, label, sample_index in zip(features, labels, sample_indices):
                sample_index = int(sample_index)
                first_frame_path = dataset.inputs[sample_index][0]
                feature_path = build_output_path(output_dir, image_root, first_frame_path)
                feature_path.parent.mkdir(parents=True, exist_ok=True)

                if feature_path.exists() and not args.overwrite:
                    skipped_count += 1
                else:
                    torch.save(
                        {
                            "features": feature,
                            "label": int(label),
                            "image_paths": dataset.inputs[sample_index],
                            "backbone_name": args.backbone_name,
                            "image_size": args.image_size,
                            "num_frames": args.num_frames,
                        },
                        feature_path,
                    )
                    saved_count += 1

                manifest.append(
                    {
                        "feature_path": str(feature_path),
                        "label": int(label),
                        "source_dir": str(Path(first_frame_path).parent),
                    }
                )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "image_root": str(image_root),
                "backbone_name": args.backbone_name,
                "image_size": args.image_size,
                "num_frames": args.num_frames,
                "samples": manifest,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {saved_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute frozen MobileNetV4 frame features for vision training."
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Image root, e.g. ../data/Training/01.원천데이터/이미지",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where feature .pt files and manifest.json are saved.",
    )
    parser.add_argument("--backbone-name", type=str, default="mobilenetv4_conv_medium")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--use-fast-decode", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    save_features(parse_args())
