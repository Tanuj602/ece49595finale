import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mpiifacegaze_dataset import MPIIFaceGazeDataset
from train_unityeyes import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune UnityEyes-pretrained gaze model on MPIIFaceGaze."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to MPIIFaceGaze root (folder with p00...p14).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/unityeyes_pretrain/best.pt"),
        help="Pretrained checkpoint to start from.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/mpiifacegaze_ft"),
        help="Directory to save fine-tuned checkpoints.",
    )
    parser.add_argument(
        "--participants",
        type=str,
        nargs="*",
        help="Optional participant IDs to include (e.g., p00 p01 p02).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable color jitter augmentation for training.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze all layers except the final regressor head.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def create_train_val_datasets(
    data_root: Path,
    participants,
    image_size: int,
    val_split: float,
    augment: bool,
    seed: int,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Build separate train/val datasets with shared index split."""

    base_train = MPIIFaceGazeDataset(
        root=data_root,
        participants=participants,
        image_size=image_size,
        augment=augment,
        eye_from_annotation=True,
    )
    base_val = MPIIFaceGazeDataset(
        root=data_root,
        participants=participants,
        image_size=image_size,
        augment=False,  # no augmentation for validation
        eye_from_annotation=True,
    )

    n = len(base_train)
    val_size = max(1, int(n * val_split))
    train_size = n - val_size

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=g)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_ds = Subset(base_train, train_idx)
    val_ds = Subset(base_val, val_idx)
    return train_ds, val_ds


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


def load_model(checkpoint: Path, device: torch.device, freeze_backbone: bool) -> nn.Module:
    model = build_model(from_scratch=False)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state
        # keep strict=False for robustness to minor key diffs
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    model.to(device)
    return model


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    train_ds, val_ds = create_train_val_datasets(
        data_root=args.data_root,
        participants=args.participants,
        image_size=args.image_size,
        val_split=args.val_split,
        augment=not args.no_augment,
        seed=args.seed,
    )

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

    model = load_model(args.checkpoint, device, freeze_backbone=args.freeze_backbone)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        ckpt = {"model": model.state_dict(), "epoch": epoch}
        (args.out_dir / "last.pt").write_bytes(torch.save(ckpt, args.out_dir / "last.pt") or b"")  # workaround for mypy
        torch.save(ckpt, args.out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, args.out_dir / "best.pt")
            print(f"  New best val loss: {val_loss:.5f} -> saved to {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
