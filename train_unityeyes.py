import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm

from unityeyes_dataset import UnityEyesDataset


def build_model(from_scratch: bool = False) -> nn.Module:
    """Return a small gaze regressor based on ResNet-18."""
    if from_scratch:
        resnet = models.resnet18(weights=None)
    else:
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, 2),  # yaw, pitch
    )
    return resnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train a gaze model on UnityEyes.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to UnityEyes dataset root.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/unityeyes_pretrain"), help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of samples for validation.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-augment", action="store_true", help="Disable basic data augmentation.")
    parser.add_argument("--from-scratch", action="store_true", help="Do not start from ImageNet weights.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_dataset(dataset: UnityEyesDataset, val_split: float, seed: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    return random_split(dataset, lengths=[train_size, val_size], generator=torch.Generator().manual_seed(seed))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    running_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        gaze = batch["gaze"].to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(images)
        loss = loss_fn(pred, gaze)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    running_loss = 0.0
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        gaze = batch["gaze"].to(device, non_blocking=True)
        pred = model(images)
        loss = loss_fn(pred, gaze)
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    dataset = UnityEyesDataset(
        root=args.data_root,
        image_size=args.image_size,
        augment=not args.no_augment,
    )
    train_ds, val_ds = split_dataset(dataset, val_split=args.val_split, seed=args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(from_scratch=args.from_scratch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        ckpt_path = args.out_dir / "last.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.out_dir / "best.pt")
            print(f"  New best val loss: {val_loss:.5f} -> saved to {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
