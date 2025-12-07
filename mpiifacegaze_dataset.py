from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from unityeyes_dataset import _vector_to_yaw_pitch


@dataclass
class _Sample:
    img_path: Path
    landmarks: np.ndarray  # shape: (6, 2)
    face_center: np.ndarray  # shape: (3,)
    gaze_target: np.ndarray  # shape: (3,)
    eye: str  # "left" or "right"


def _build_transform(image_size: int, augment: bool) -> T.Compose:
    aug: List[torch.nn.Module] = []
    if augment:
        aug.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02))

    aug.extend(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return T.Compose(aug)


def _parse_annotation_line(participant_root: Path, line: str) -> _Sample:
    parts = line.strip().split()
    if len(parts) < 28:
        raise ValueError(f"Malformed annotation line in {participant_root.name}: '{line[:80]}'")

    img_rel = parts[0]
    numbers = [float(p) for p in parts[1:-1]]
    eye = parts[-1].lower()

    if len(numbers) < 26:
        raise ValueError(f"Expected at least 26 numeric fields, got {len(numbers)} in line: {line[:80]}")

    # Numbers layout: [screen_x, screen_y, 12 landmarks, 6 head pose, 3 face center, 3 gaze target]
    landmarks = np.asarray(numbers[2:14], dtype=np.float32).reshape(6, 2)
    face_center = np.asarray(numbers[20:23], dtype=np.float32)
    gaze_target = np.asarray(numbers[23:26], dtype=np.float32)

    img_path = participant_root / img_rel
    return _Sample(img_path=img_path, landmarks=landmarks, face_center=face_center, gaze_target=gaze_target, eye=eye)


def _select_eye_corners(landmarks: np.ndarray, eye: str) -> np.ndarray:
    if eye == "left":
        return landmarks[0:2]
    if eye == "right":
        return landmarks[2:4]
    # Fallback: use both eye corners
    return landmarks[0:4]


def _crop_eye(image: Image.Image, landmarks: np.ndarray, eye: str, scale: float = 2.2) -> Image.Image:
    eye_pts = _select_eye_corners(landmarks, eye)
    eye_pts = np.asarray(eye_pts, dtype=np.float32).reshape(-1, 2)
    center = eye_pts.mean(axis=0)

    if eye_pts.shape[0] >= 2:
        width = float(np.linalg.norm(eye_pts[0] - eye_pts[-1]))
    else:
        width = 60.0
    box_size = max(32.0, width * scale)

    cx, cy = center.tolist()
    half = box_size * 0.5
    left = int(max(0, cx - half))
    top = int(max(0, cy - half))
    right = int(min(image.width, cx + half))
    bottom = int(min(image.height, cy + half))

    return image.crop((left, top, right, bottom))


class MPIIFaceGazeDataset(Dataset):
    """PyTorch dataset for MPIIFaceGaze.

    Each sample crops the requested eye region and returns yaw/pitch gaze angles in radians.
    """

    def __init__(
        self,
        root: Path | str,
        participants: Optional[Sequence[str]] = None,
        image_size: int = 224,
        augment: bool = False,
        eye_from_annotation: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"MPIIFaceGaze root does not exist: {self.root}")

        self.participants = self._discover_participants(participants)
        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(f"No samples found under {self.root} for participants {self.participants}")

        self.transform = _build_transform(image_size=image_size, augment=augment)
        self.eye_from_annotation = eye_from_annotation

    def _discover_participants(self, participants: Optional[Sequence[str]]) -> List[str]:
        if participants:
            return [p for p in participants]
        return sorted([p.name for p in self.root.iterdir() if p.is_dir() and p.name.startswith("p")])

    def _load_samples(self) -> List[_Sample]:
        samples: List[_Sample] = []
        for pid in self.participants:
            participant_root = self.root / pid
            annot_path = participant_root / f"{pid}.txt"
            if not annot_path.exists():
                continue
            with annot_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        samples.append(_parse_annotation_line(participant_root, line))
                    except ValueError:
                        # Skip malformed lines
                        continue
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.img_path).convert("RGB")

        eye_side = sample.eye if self.eye_from_annotation else "left"
        cropped = _crop_eye(image, sample.landmarks, eye_side)

        gaze_vec = sample.gaze_target - sample.face_center
        yaw, pitch = _vector_to_yaw_pitch(gaze_vec)

        image_t = self.transform(cropped)
        gaze = torch.tensor([float(yaw), float(pitch)], dtype=torch.float32)

        return {"image": image_t, "gaze": gaze, "path": str(sample.img_path)}
