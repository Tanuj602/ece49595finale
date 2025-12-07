import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

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
# Calibration target helper
# -----------------------------
def make_targets(screen_w: int, screen_h: int, n_points: int) -> list[tuple[int, int]]:
    margin_x = int(screen_w * 0.05)
    margin_y = int(screen_h * 0.05)
    xs = [margin_x, screen_w // 2, screen_w - margin_x]
    ys = [margin_y, screen_h // 2, screen_h - margin_y]
    grid9 = [(x, y) for y in ys for x in xs]
    if n_points <= 9:
        return grid9[:n_points]

    mid_edges = [
        (screen_w // 2, margin_y),
        (screen_w // 2, screen_h - margin_y),
        (margin_x, screen_h // 2),
        (screen_w - margin_x, screen_h // 2),
    ]
    targets = grid9 + mid_edges
    if n_points <= len(targets):
        return targets[:n_points]

    # Add inner quadrant points
    qx1 = int(screen_w * 0.33)
    qx2 = int(screen_w * 0.67)
    qy1 = int(screen_h * 0.33)
    qy2 = int(screen_h * 0.67)
    inner = [(qx1, qy1), (qx2, qy1), (qx1, qy2), (qx2, qy2)]
    targets += inner
    return targets[:n_points]


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
    ap.add_argument("--screen-width", type=int, default=1920)
    ap.add_argument("--screen-height", type=int, default=1080)
    ap.add_argument("--calib-points", type=int, default=12)
    ap.add_argument("--calib-settle-ms", type=int, default=400)
    ap.add_argument("--calib-collect-ms", type=int, default=900)
    ap.add_argument("--smoothing", type=float, default=0.25, help="EMA on yaw/pitch; 0 disables smoothing")
    ap.add_argument("--window-width", type=int, default=1280)
    ap.add_argument("--window-height", type=int, default=720)
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

    screen_w, screen_h = args.screen_width, args.screen_height
    targets = make_targets(screen_w, screen_h, args.calib_points)

    # Calibration state
    A = None
    auto_active = False
    auto_phase = "idle"  # idle | settle | collect | done
    auto_idx = 0
    phase_start_t = time.time()
    calib_yps: list[tuple[float, float]] = []
    calib_xys: list[tuple[float, float]] = []
    prev_yp = None

    print("Controls:")
    print("  q : quit")
    print("  c : start auto-calibration (follow the red dot)")
    print("  r : reset calibration")

    mini_w, mini_h = 220, 160

    cv2.namedWindow("Gaze Demo (UnityEyes -> MPIIFaceGaze)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Demo (UnityEyes -> MPIIFaceGaze)", args.window_width, args.window_height)

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

        # EMA smoothing
        if yaw_pred is not None:
            if prev_yp is None or args.smoothing <= 0:
                prev_yp = (yaw_pred, pitch_pred)
            else:
                a = float(np.clip(args.smoothing, 0.0, 1.0))
                prev_yp = (
                    a * yaw_pred + (1 - a) * prev_yp[0],
                    a * pitch_pred + (1 - a) * prev_yp[1],
                )
        yp_for_map = prev_yp

        # Auto calibration state machine
        if auto_active:
            tgt = targets[auto_idx] if auto_idx < len(targets) else None
            elapsed_ms = (time.time() - phase_start_t) * 1000.0

            if tgt is None:
                if len(calib_yps) >= 6:
                    A = fit_affine(calib_yps, calib_xys)
                    print("Auto calibration complete. Samples:", len(calib_yps))
                else:
                    print("Auto calibration ended with too few samples.")
                auto_active = False
                auto_phase = "done"
            else:
                if auto_phase == "settle":
                    if elapsed_ms >= args.calib_settle_ms:
                        auto_phase = "collect"
                        phase_start_t = time.time()
                elif auto_phase == "collect":
                    if yp_for_map is not None:
                        calib_yps.append(yp_for_map)
                        calib_xys.append(tgt)
                    if elapsed_ms >= args.calib_collect_ms:
                        auto_idx += 1
                        auto_phase = "settle"
                        phase_start_t = time.time()

        # If we have a mapping, compute predicted screen coords
        pred_x = pred_y = None
        if A is not None and yp_for_map is not None:
            pred_x, pred_y = apply_affine(A, yp_for_map[0], yp_for_map[1])

        # Overlay text
        y0 = 25
        if yp_for_map is not None:
            cv2.putText(frame, f"yaw={yp_for_map[0]:.3f} rad  pitch={yp_for_map[1]:.3f} rad",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y0 += 24
        else:
            cv2.putText(frame, "No face/eyes detected",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y0 += 24

        if auto_active:
            cv2.putText(frame, f"CALIBRATING {auto_idx+1}/{len(targets)} phase={auto_phase}",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y0 += 24
        elif A is not None:
            cv2.putText(frame, "Calibrated", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y0 += 24
        else:
            cv2.putText(frame, "Not calibrated (press c)", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y0 += 24

        # Full-frame overlay for target/prediction (fills the window)
        if screen_w and screen_h:
            if auto_active and auto_idx < len(targets):
                tx, ty = targets[auto_idx]
                ux = np.clip(tx / screen_w, 0, 1)
                uy = np.clip(ty / screen_h, 0, 1)
                cx = int(ux * w)
                cy = int(uy * h)
                cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)

            if pred_x is not None:
                ux = np.clip(pred_x / screen_w, 0, 1)
                uy = np.clip(pred_y / screen_h, 0, 1)
                gx = int(ux * w)
                gy = int(uy * h)
                cv2.drawMarker(frame, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
                cv2.circle(frame, (gx, gy), 10, (0, 255, 0), 2)

        # Mini screen map (top-right) for reference
        x1 = w - mini_w - 10
        y1 = 10
        x2 = x1 + mini_w
        y2 = y1 + mini_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        if auto_active and auto_idx < len(targets):
            tx, ty = targets[auto_idx]
            ux = np.clip(tx / screen_w, 0, 1)
            uy = np.clip(ty / screen_h, 0, 1)
            cx = int(x1 + ux * mini_w)
            cy = int(y1 + uy * mini_h)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        if pred_x is not None:
            ux = np.clip(pred_x / screen_w, 0, 1)
            uy = np.clip(pred_y / screen_h, 0, 1)
            gx = int(x1 + ux * mini_w)
            gy = int(y1 + uy * mini_h)
            cv2.circle(frame, (gx, gy), 6, (0, 255, 0), -1)

        cv2.imshow("Gaze Demo (UnityEyes -> MPIIFaceGaze)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            auto_active = True
            auto_phase = "settle"
            auto_idx = 0
            phase_start_t = time.time()
            calib_yps.clear()
            calib_xys.clear()
            A = None
            print("Auto calibration started. Follow the red dot.")
        elif key == ord("r"):
            print("Calibration reset.")
            A = None
            auto_active = False
            auto_phase = "idle"
            calib_yps.clear()
            calib_xys.clear()
            prev_yp = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
