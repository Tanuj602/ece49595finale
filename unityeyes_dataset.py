import json
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


def _default_transform(image_size: int, augment: bool) -> T.Compose:
    """Create basic image transforms for UnityEyes samples.

    Note: we deliberately avoid horizontal flips here. Flipping the image
    would require flipping the yaw label as well; since we're not doing
    label-aware transforms in this module, we keep the geometry fixed.
    """
    aug: List[torch.nn.Module] = []
    if augment:
        aug.append(
            T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
                hue=0.02,
            )
        )

    aug.extend(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return T.Compose(aug)


def _parse_look_vec(raw: object) -> Optional[List[float]]:
    """Normalize different UnityEyes look_vec formats into a float list."""
    if raw is None:
        return None
    if isinstance(raw, str):
        # Strings often look like "(0.8372, 0.1541, -0.5248, 0.0000)"
        cleaned = (
            raw.strip()
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
        )
        parts = [p for p in cleaned.replace(",", " ").split() if p]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return None
    if isinstance(raw, Iterable):
        try:
            return [float(x) for x in list(raw)]
        except (TypeError, ValueError):
            return None
    return None


def _vector_to_yaw_pitch(vec: Sequence[float]) -> Tuple[float, float]:
    """
    Convert a 3D gaze vector to yaw/pitch in radians.

    UnityEyes uses a camera-forward -Z convention; yaw is positive to the left,
    pitch is positive up.
    """
    v = np.asarray(vec[:3], dtype=np.float32)
    norm = np.linalg.norm(v) + 1e-8
    v /= norm
    yaw = math.atan2(v[0], -v[2])
    pitch = math.asin(np.clip(v[1], -1.0, 1.0))
    return yaw, pitch


def _extract_look_vec(label: dict) -> Optional[List[float]]:
    """Try multiple common UnityEyes label keys to find a gaze direction vector."""
    candidates = [
        label.get("eye_details", {}).get("look_vec"),
        label.get("look_vec"),
        label.get("gaze_vector"),
        label.get("gaze", {}).get("vector"),
        label.get("gazevector"),
    ]
    for c in candidates:
        vec = _parse_look_vec(c)
        if vec:
            return vec
    return None


class UnityEyesDataset(Dataset):
    """
    Thin PyTorch dataset for UnityEyes synthetic eye images.

    Expected structure: image files (.png/.jpg) with matching JSON label files
    of the same stem. Each JSON should contain a 3D gaze direction vector
    (e.g., label["eye_details"]["look_vec"]).
    """

    def __init__(
        self,
        root: Path | str,
        image_size: int = 224,
        augment: bool = False,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"UnityEyes root does not exist: {self.root}")

        self.samples = self._collect_samples(self.root, extensions)
        if not self.samples:
            raise RuntimeError(
                f"No UnityEyes samples with labels found under {self.root}. "
                "Expected image files with matching JSON label files."
            )

        self.transform = _default_transform(image_size=image_size, augment=augment)

    @staticmethod
    def _collect_samples(root: Path, extensions: Tuple[str, ...]) -> List[Tuple[Path, Path]]:
        samples: List[Tuple[Path, Path]] = []
        for ext in extensions:
            for img_path in root.rglob(f"*{ext}"):
                label_path = img_path.with_suffix(".json")
                if label_path.exists():
                    samples.append((img_path, label_path))
        samples.sort(key=lambda p: str(p[0]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_label(self, label_path: Path) -> Tuple[float, float]:
        with label_path.open("r", encoding="utf-8") as f:
            label = json.load(f)
        look_vec = _extract_look_vec(label)
        if look_vec is None or len(look_vec) < 3:
            raise KeyError(f"No gaze vector found in {label_path}")
        yaw, pitch = _vector_to_yaw_pitch(look_vec)
        return float(yaw), float(pitch)

    def __getitem__(self, idx: int):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        yaw, pitch = self._read_label(label_path)

        image_t = self.transform(image)
        gaze = torch.tensor([yaw, pitch], dtype=torch.float32)
        return {
            "image": image_t,
            "gaze": gaze,
            "path": str(img_path),
        }
