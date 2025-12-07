import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image

# Optional (for showing cursor + screen mapping)
try:
    import pyautogui
    HAS_PYAUTOGUI = True
except Exception:
    HAS_PYAUTOGUI = False

# MediaPipe for face/eye landmarks
import mediapipe as mp

# Use your exact model definition
from train_unityeyes import build_model


# -----------------------------
# Preprocess (match your datasets)
# -----------------------------
def build_preprocess(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def preprocess_crop_bgr(crop_bgr, tfm):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    x = tfm(pil).unsqueeze(0)
    return x


# -----------------------------
# Eye crop helper (mirrors your MPII crop logic)
# -----------------------------
def crop_eye_from_corners(image_bgr, p1, p2, scale=2.2):
    """
    p1, p2: (x,y) pixel coords of eye corners
    Uses similar logic to _crop_eye in your dataset:
    center = mean(corners)
    width = distance between corners
    box_size = max(32, width * scale)
    """
    h, w = image_bgr.shape[:2]
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    center = (p1 + p2) * 0.5
    width = float(np.linalg.norm(p1 - p2))
    box_size = max(32.0, width * scale)

    cx, cy = center.tolist()
    half = box_size * 0.5

    left   = int(max(0, cx - half))
    top    = int(max(0, cy - half))
    right  = int(min(w, cx + half))
    bottom = int(min(h, cy + half))

    if right <= left or bottom <= top:
        return None, None

    crop = image_bgr[top:bottom, left:right].copy()
    return crop, (left, top, right, bottom)


# -----------------------------
# Simple affine calibration:
# screen_xy ~= A * [yaw, pitch, 1]
# -----------------------------
def fit_affine(yaw_pitch_list, screen_xy_list):
    X = np.array([[yp[0], yp[1], 1.0] for yp in yaw_pitch_list], dtype=np.float32)  # N x 3
    Y = np.array(screen_xy_list, dtype=np.float32)  # N x 2
    # Solve least squares: X * A = Y  -> A is 3 x 2
    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return A  # 3x2


def apply_affine(A, yaw, pitch):
    vec = np.array([yaw, pitch, 1.0], dtype=np.float32)
    xy = vec @ A
    return float(xy[0]), float(xy[1])


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/mpiifacegaze_ft/best.pt"))
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--eye", type=str, default="both", choices=["left", "right", "both"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-cursor", action="store_true", help="Disable cursor display/mapping even if pyautogui is available.")
    return ap.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    model = build_model(from_scratch=False)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    tfm = build_preprocess(args.image_size)

    # MediaPipe Face Mesh setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # includes iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # MediaPipe eye corner indices
    # Left eye: outer=33, inner=133
    # Right eye: outer=362, inner=263
    L_OUT, L_IN = 33, 133
    R_OUT, R_IN = 362, 263

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Cursor/screen helper
    use_cursor = (HAS_PYAUTOGUI and (not args.no_cursor))
    if use_cursor:
        screen_w, screen_h = pyautogui.size()
        print(f"Screen: {screen_w} x {screen_h}")
    else:
        screen_w, screen_h = None, None

    # Calibration storage
    A = None
    collecting = False
    calib_yps = []
    calib_xys = []
    calib_target_n = 60  # ~2 seconds worth at ~30fps

    print("Controls:")
    print("  q  : quit")
    print("  c  : start cursor-based calibration (look at your cursor and move it around)")
    print("  r  : reset calibration")

    # Mini-map size for screen visualization
    mini_w, mini_h = 220, 160

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        yaw_pred = None
        pitch_pred = None
        eye_boxes = []

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            def to_px(idx):
                return int(lm[idx].x * w), int(lm[idx].y * h)

            l_out = to_px(L_OUT)
            l_in  = to_px(L_IN)
            r_out = to_px(R_OUT)
            r_in  = to_px(R_IN)

            preds = []

            if args.eye in ("left", "both"):
                crop, box = crop_eye_from_corners(frame, l_out, l_in, scale=2.2)
                if crop is not None:
                    x = preprocess_crop_bgr(crop, tfm).to(device)
                    with torch.no_grad():
                        yp = model(x).cpu().numpy().reshape(-1)
                    preds.append(yp)
                    eye_boxes.append(("L", box))

            if args.eye in ("right", "both"):
                crop, box = crop_eye_from_corners(frame, r_out, r_in, scale=2.2)
                if crop is not None:
                    x = preprocess_crop_bgr(crop, tfm).to(device)
                    with torch.no_grad():
                        yp = model(x).cpu().numpy().reshape(-1)
                    preds.append(yp)
                    eye_boxes.append(("R", box))

            if preds:
                yp_avg = np.mean(np.stack(preds, axis=0), axis=0)
                yaw_pred, pitch_pred = float(yp_avg[0]), float(yp_avg[1])

        # Draw eye boxes
        for tag, box in eye_boxes:
            if box:
                l, t, r, b = box
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 1)
                cv2.putText(frame, tag, (l, t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Cursor position
        mouse_x = mouse_y = None
        if use_cursor:
            try:
                mouse_x, mouse_y = pyautogui.position()
            except Exception:
                mouse_x = mouse_y = None

        # If calibrating, collect pairs (yaw,pitch) <-> cursor
        if collecting and yaw_pred is not None and mouse_x is not None:
            calib_yps.append((yaw_pred, pitch_pred))
            calib_xys.append((mouse_x, mouse_y))

            if len(calib_yps) >= calib_target_n:
                A = fit_affine(calib_yps, calib_xys)
                collecting = False
                print("Calibration complete. Affine mapping learned.")
                print("  Samples used:", len(calib_yps))

        # If we have a mapping, compute predicted screen coords
        pred_x = pred_y = None
        err_px = None
        if A is not None and yaw_pred is not None:
            pred_x, pred_y = apply_affine(A, yaw_pred, pitch_pred)
            if mouse_x is not None:
                dx = pred_x - mouse_x
                dy = pred_y - mouse_y
                err_px = (dx * dx + dy * dy) ** 0.5

        # Overlay text
        y0 = 25
        if yaw_pred is not None:
            cv2.putText(frame, f"yaw={yaw_pred:.3f} rad  pitch={pitch_pred:.3f} rad",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y0 += 24
        else:
            cv2.putText(frame, "No face/eyes detected",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y0 += 24

        if use_cursor and mouse_x is not None:
            cv2.putText(frame, f"cursor=({mouse_x},{mouse_y})",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y0 += 24

        if pred_x is not None:
            cv2.putText(frame, f"pred  =({int(pred_x)},{int(pred_y)})",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y0 += 24
            if err_px is not None:
                cv2.putText(frame, f"err   ={err_px:.1f} px",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y0 += 24

        if collecting:
            cv2.putText(frame, f"CALIBRATING... {len(calib_yps)}/{calib_target_n}",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Mini screen map (top-right)
        if use_cursor:
            x1 = w - mini_w - 10
            y1 = 10
            x2 = x1 + mini_w
            y2 = y1 + mini_h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

            if mouse_x is not None and screen_w:
                ux = mouse_x / screen_w
                uy = mouse_y / screen_h
                mx = int(x1 + ux * mini_w)
                my = int(y1 + uy * mini_h)
                cv2.circle(frame, (mx, my), 4, (0, 0, 255), -1)  # red cursor

            if pred_x is not None and screen_w:
                ux = np.clip(pred_x / screen_w, 0, 1)
                uy = np.clip(pred_y / screen_h, 0, 1)
                gx = int(x1 + ux * mini_w)
                gy = int(y1 + uy * mini_h)
                cv2.circle(frame, (gx, gy), 4, (0, 255, 0), -1)  # green gaze

        cv2.imshow("Gaze Demo (UnityEyes -> MPIIFaceGaze)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            if not use_cursor:
                print("pyautogui not available. Install it or run without --no-cursor.")
            else:
                print("Starting calibration: look at your cursor and move it around.")
                collecting = True
                calib_yps.clear()
                calib_xys.clear()
        elif key == ord("r"):
            print("Calibration reset.")
            A = None
            collecting = False
            calib_yps.clear()
            calib_xys.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
