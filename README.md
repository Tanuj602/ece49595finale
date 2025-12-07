# UnityEyes Pretraining Scaffold

This repo contains a minimal first step to pre-train a gaze regressor on UnityEyes synthetic images before adapting to MPIIFaceGaze or other real-world data.

## Setup
- Python 3.10+ recommended.
- Install dependencies: `python3 -m pip install -r requirements.txt`

## UnityEyes data layout
- Place the UnityEyes dataset locally (not included here).
- The loader expects image files (`.png`/`.jpg`) with matching JSON labels of the same stem:
  - `some_dir/img_000001.png`
  - `some_dir/img_000001.json` containing a gaze vector (e.g., `eye_details.look_vec`).
- The code scans all subfolders of `--data-root` for this pattern.

## Run pretraining
Example run (3 quick epochs to sanity-check the pipeline):
```bash
python3 train_unityeyes.py \
  --data-root /path/to/UnityEyes \
  --out-dir runs/unityeyes_pretrain \
  --epochs 3 \
  --batch-size 64
```

Important flags:
- `--no-augment` to disable color jitter and flips.
- `--from-scratch` to avoid ImageNet initialization (by default it uses ResNet-18 ImageNet weights).
- `--cpu` to force CPU if CUDA is present.

## Outputs
- `runs/unityeyes_pretrain/last.pt`: latest epoch weights.
- `runs/unityeyes_pretrain/best.pt`: best validation loss weights.

## Next steps
- Increase `--epochs` and tweak `--val-split` once you confirm labels are read correctly.
- Add a short per-user calibration layer or fine-tune on MPIIFaceGaze after this pretraining.
- Consider domain randomization (occlusion, blur, lighting) if UnityEyes → real-world transfer is poor.
