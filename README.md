# Gaze Regression: UnityEyes → MPIIFaceGaze

This repo pre-trains a ResNet-18 gaze regressor on UnityEyes and fine-tunes it on MPIIFaceGaze, plus a webcam demo for quick sanity checks.

## Setup
- Python 3.10+ recommended.
- Install deps:
  ```bash
  python -m pip install -r requirements.txt
  python -m pip install opencv-python
  ```

## Data layout
**UnityEyes**
- Place UnityEyes locally (not included).
- Expects image files (`.png`/`.jpg`) with matching JSON labels of the same stem:
  - `some_dir/img_000001.png`
  - `some_dir/img_000001.json` containing a gaze vector (e.g., `eye_details.look_vec`).
- The loader scans all subfolders of `--data-root`.

**MPIIFaceGaze**
- Root contains participants `p00` … `p14`.
- Each `pXX` has many `dayYY` image folders and an annotation file `pXX.txt`.
- We use the annotation fields: eye-corner landmarks (for cropping), face center, gaze target (to form a 3D gaze vector), and the eye side flag (`left`/`right`).

## Pre-train on UnityEyes
Quick sanity run:
```bash
python train_unityeyes.py \
  --data-root /path/to/UnityEyes \
  --out-dir runs/unityeyes_pretrain \
  --epochs 3 \
  --batch-size 64
```
Key flags: `--no-augment`, `--from-scratch`, `--cpu`, `--val-split`.

## Fine-tune on MPIIFaceGaze
```bash
python finetune_mpiifacegaze.py \
  --data-root MPIIFaceGaze/MPIIFaceGaze \
  --checkpoint runs/unityeyes_pretrain/best.pt \
  --out-dir runs/mpiifacegaze_ft \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4
```
Helpful flags:
- `--participants p00 p01 ...` to subset subjects.
- `--freeze-backbone` to train only the final regressor head.
- `--no-augment` to disable color jitter.
- `--cpu` to force CPU.

## Outputs
- `runs/unityeyes_pretrain/{last.pt,best.pt}` from pretraining.
- `runs/mpiifacegaze_ft/{last.pt,best.pt}` from fine-tuning.

## Webcam demo (dual-eye, coarse screen projection)
```bash
python webcam_demo.py \
  --checkpoint runs/mpiifacegaze_ft/best.pt \
  --screen-width 1920 --screen-height 1080 --focal-px 900 \
  --window-width 1600 --window-height 900
```
- Detects up to two eyes with a Haar cascade, crops, runs the model, and overlays yaw/pitch.
- Projects yaw/pitch onto the whole frame as a red dot using a simple pinhole approximation; tune `--focal-px` and make sure screen width/height match your display. (For accurate screen mapping, add head-pose or a short calibration routine.)
- Press `q` to quit; window is resizable.

## Fullscreen overlay with auto-calibration
```bash
python gaze_overlay_fullscreen.py \
  --checkpoint runs/mpiifacegaze_ft/best.pt \
  --device cuda --eye both
```
- Uses MediaPipe FaceMesh to crop both eyes, averages predictions, and renders a transparent fullscreen dot overlay.
- Built-in auto-calibration (`c` to start, follow the moving dot) fits an affine map from yaw/pitch to screen XY and smooths outputs; `r` to reset, `q` to quit.
- Good for quick on-screen gaze visualization; relies on webcam + face mesh quality.

## Tips / next steps
- Run a longer pretrain/fine-tune once you trust the pipeline; adjust `--val-split` for your hardware.
- For better real-world mapping, add a brief per-user calibration (collect a few on-screen targets and fit a small regressor on top of the yaw/pitch outputs).
- If Haar eye detection struggles, improve lighting, move closer, or swap in a face/landmark detector.
