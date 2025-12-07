import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

import mediapipe as mp

from PyQt5 import QtCore, QtGui, QtWidgets

from train_unityeyes import build_model


# -----------------------------
# Preprocess (match training)
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
# Eye crop helper (mirrors MPII logic)
# -----------------------------
def crop_eye_from_corners(image_bgr, p1, p2, scale=2.2):
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
        return None

    return image_bgr[top:bottom, left:right].copy()


# -----------------------------
# Simple affine calibration:
# screen_xy ~= A * [yaw, pitch, 1]
# -----------------------------
def fit_affine(yaw_pitch_list, screen_xy_list):
    X = np.array([[yp[0], yp[1], 1.0] for yp in yaw_pitch_list], dtype=np.float32)  # N x 3
    Y = np.array(screen_xy_list, dtype=np.float32)  # N x 2
    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return A  # 3x2


def apply_affine(A, yaw, pitch):
    vec = np.array([yaw, pitch, 1.0], dtype=np.float32)
    xy = vec @ A
    return float(xy[0]), float(xy[1])


# -----------------------------
# Gaze inference worker
# -----------------------------
class GazeWorker:
    def __init__(self, checkpoint: Path, image_size=224, device_str="cuda",
                 eye="both", camera_index=0, show_debug=False):
        self.image_size = image_size
        self.eye = eye
        self.camera_index = camera_index
        self.show_debug = show_debug

        self.device = torch.device("cpu" if device_str == "cpu" or not torch.cuda.is_available() else "cuda")

        self.model = build_model(from_scratch=False)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            self.model.load_state_dict(state["model"], strict=False)
        else:
            self.model.load_state_dict(state, strict=False)

        self.model.to(self.device)
        self.model.eval()

        self.tfm = build_preprocess(image_size)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Eye corner indices
        self.L_OUT, self.L_IN = 33, 133
        self.R_OUT, self.R_IN = 362, 263

        self.cap = None
        self.thread = None
        self.stop_flag = False

        self.lock = threading.Lock()
        self.latest_yaw_pitch = None
        self.latest_frame = None  # for debug

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        self.stop_flag = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def get_latest(self):
        with self.lock:
            return self.latest_yaw_pitch

    def _loop(self):
        while not self.stop_flag:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)

            yaw_pitch = None

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark

                def to_px(idx):
                    return int(lm[idx].x * w), int(lm[idx].y * h)

                l_out = to_px(self.L_OUT)
                l_in  = to_px(self.L_IN)
                r_out = to_px(self.R_OUT)
                r_in  = to_px(self.R_IN)

                preds = []

                if self.eye in ("left", "both"):
                    crop = crop_eye_from_corners(frame, l_out, l_in, scale=2.2)
                    if crop is not None:
                        x = preprocess_crop_bgr(crop, self.tfm).to(self.device)
                        with torch.no_grad():
                            yp = self.model(x).cpu().numpy().reshape(-1)
                        preds.append(yp)

                if self.eye in ("right", "both"):
                    crop = crop_eye_from_corners(frame, r_out, r_in, scale=2.2)
                    if crop is not None:
                        x = preprocess_crop_bgr(crop, self.tfm).to(self.device)
                        with torch.no_grad():
                            yp = self.model(x).cpu().numpy().reshape(-1)
                        preds.append(yp)

                if preds:
                    yp_avg = np.mean(np.stack(preds, axis=0), axis=0)
                    yaw_pitch = (float(yp_avg[0]), float(yp_avg[1]))

            with self.lock:
                self.latest_yaw_pitch = yaw_pitch
                self.latest_frame = frame

            if self.show_debug:
                dbg = frame.copy()
                if yaw_pitch is not None:
                    cv2.putText(dbg, f"yaw={yaw_pitch[0]:.3f}, pitch={yaw_pitch[1]:.3f}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Gaze Debug Camera", dbg)
                cv2.waitKey(1)

            time.sleep(0.001)


# -----------------------------
# Overlay window with AUTO calibration
# -----------------------------
class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, worker: GazeWorker, calib_points=12,
                 settle_ms=400, collect_ms=900, smoothing=0.25):
        super().__init__()
        self.worker = worker

        # Screen size from Qt (more reliable than pyautogui for DPI)
        screen = QtWidgets.QApplication.primaryScreen()
        geo = screen.geometry()
        self.screen_w = geo.width()
        self.screen_h = geo.height()

        # Calibration mapping
        self.A = None

        # Auto calibration state
        self.auto_active = False
        self.auto_targets = self._make_targets(calib_points)
        self.auto_idx = 0
        self.auto_phase = "idle"  # idle | settle | collect | done
        self.phase_start_t = time.monotonic()

        self.settle_ms = int(settle_ms)
        self.collect_ms = int(collect_ms)

        self.calib_yps = []
        self.calib_xys = []

        # Dot state
        self.pred_x = None
        self.pred_y = None

        # Exponential smoothing
        self.smoothing = float(smoothing)

        # UI config
        self.setWindowTitle("Gaze Overlay")
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        self.setGeometry(0, 0, self.screen_w, self.screen_h)
        self.showFullScreen()

        # Update loop ~60Hz
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_state)
        self.timer.start(16)

        print("Overlay console commands:")
        print("  c : start AUTO calibration (follow the red dot)")
        print("  r : reset calibration")
        print("  q : quit")

    def _make_targets(self, n_points: int):
        """
        Build a balanced set of calibration points.
        Default pattern:
          - 3x3 grid (9 points)
          - plus extra mid-edge points for 12
          - plus corners+center+edges for 16
        """
        margin_x = int(self.screen_w * 0.02)
        margin_y = int(self.screen_h * 0.02)
        xs = [margin_x, self.screen_w // 2, self.screen_w - margin_x]
        ys = [margin_y, self.screen_h // 2, self.screen_h - margin_y]

        grid9 = [(x, y) for y in ys for x in xs]  # row-major

        if n_points <= 9:
            return grid9[:n_points]

        # Add mid-edge points (top, bottom, left, right)
        mid_edges = [
            (self.screen_w // 2, margin_y),
            (self.screen_w // 2, self.screen_h - margin_y),
            (margin_x, self.screen_h // 2),
            (self.screen_w - margin_x, self.screen_h // 2),
        ]

        targets = grid9 + mid_edges

        if n_points <= 13:
            return targets[:n_points]

        # Add 4 additional inner-quadrant points for 16-ish
        qx1 = int(self.screen_w * 0.33)
        qx2 = int(self.screen_w * 0.67)
        qy1 = int(self.screen_h * 0.33)
        qy2 = int(self.screen_h * 0.67)
        inner = [(qx1, qy1), (qx2, qy1), (qx1, qy2), (qx2, qy2)]

        targets += inner

        return targets[:n_points]

    def start_auto_calibration(self):
        self.auto_active = True
        self.auto_idx = 0
        self.auto_phase = "settle"
        self.phase_start_t = time.monotonic()
        self.calib_yps.clear()
        self.calib_xys.clear()
        self.pred_x = self.pred_y = None
        self.A = None
        print(f"Auto calibration started with {len(self.auto_targets)} targets.")
        print(f"  settle per target: {self.settle_ms} ms")
        print(f"  collect per target: {self.collect_ms} ms")

    def reset_calibration(self):
        self.A = None
        self.auto_active = False
        self.auto_phase = "idle"
        self.calib_yps.clear()
        self.calib_xys.clear()
        self.pred_x = self.pred_y = None
        print("Calibration reset.")

    def _current_target(self):
        if not self.auto_targets or self.auto_idx >= len(self.auto_targets):
            return None
        return self.auto_targets[self.auto_idx]

    def update_state(self):
        yp = self.worker.get_latest()
        now = time.monotonic()

        # ------------ AUTO CALIBRATION STATE MACHINE ------------
        if self.auto_active:
            tgt = self._current_target()

            if tgt is None:
                # finalize
                if len(self.calib_yps) >= 10:
                    self.A = fit_affine(self.calib_yps, self.calib_xys)
                    print("Auto calibration complete.")
                    print("  total samples:", len(self.calib_yps))
                else:
                    print("Auto calibration ended with too few samples.")
                self.auto_active = False
                self.auto_phase = "done"
                self.update()
                return

            elapsed_ms = (now - self.phase_start_t) * 1000.0

            if self.auto_phase == "settle":
                if elapsed_ms >= self.settle_ms:
                    self.auto_phase = "collect"
                    self.phase_start_t = now

            elif self.auto_phase == "collect":
                if yp is not None:
                    yaw, pitch = yp
                    self.calib_yps.append((yaw, pitch))
                    self.calib_xys.append(tgt)

                if elapsed_ms >= self.collect_ms:
                    # next target
                    self.auto_idx += 1
                    self.auto_phase = "settle"
                    self.phase_start_t = now

            # While calibrating we do NOT update live gaze dot
            self.update()
            return

        # ------------ NORMAL LIVE PREDICTION ------------
        if yp is None:
            self.update()
            return

        yaw, pitch = yp

        if self.A is not None:
            x, y = apply_affine(self.A, yaw, pitch)

            x = max(0.0, min(self.screen_w - 1.0, x))
            y = max(0.0, min(self.screen_h - 1.0, y))

            if self.pred_x is None:
                self.pred_x, self.pred_y = x, y
            else:
                a = self.smoothing
                self.pred_x = (1 - a) * self.pred_x + a * x
                self.pred_y = (1 - a) * self.pred_y + a * y

        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # AUTO CALIBRATION visuals
        if self.auto_active:
            self._draw_auto_calibration_ui(painter)
            return

        # Not calibrated hint
        if self.A is None:
            self._draw_hint(painter)
            return

        # Live gaze dot
        if self.pred_x is not None:
            # Outer halo
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 180))
            pen.setWidth(12)
            painter.setPen(pen)
            painter.drawPoint(int(self.pred_x), int(self.pred_y))

            # Inner bright dot
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0, 230))
            pen.setWidth(7)
            painter.setPen(pen)
            painter.drawPoint(int(self.pred_x), int(self.pred_y))

    def _draw_auto_calibration_ui(self, painter):
        # Draw instruction box
        rect = QtCore.QRect(20, 20, 560, 150)
        painter.setBrush(QtGui.QColor(0, 0, 0, 140))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(rect, 10, 10)

        painter.setPen(QtGui.QColor(255, 255, 255, 230))
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)

        total = len(self.auto_targets)
        idx = min(self.auto_idx + 1, total)
        msg = (
            "Calibration in progress\n"
            "Follow the RED dot with your eyes.\n"
            "Try to keep your head still.\n"
            f"Target {idx}/{total}"
        )
        painter.drawText(rect.adjusted(12, 10, -12, -10),
                         QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, msg)

        # Draw the current target dot
        tgt = self._current_target()
        if tgt is None:
            return
        x, y = tgt

        # Big red ring + white core for visibility
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 200))
        pen.setWidth(14)
        painter.setPen(pen)
        painter.drawPoint(int(x), int(y))

        pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 240))
        pen.setWidth(10)
        painter.setPen(pen)
        painter.drawPoint(int(x), int(y))

        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 240))
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawPoint(int(x), int(y))

    def _draw_hint(self, painter):
        rect = QtCore.QRect(20, 20, 560, 140)
        painter.setBrush(QtGui.QColor(0, 0, 0, 120))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(rect, 10, 10)

        painter.setPen(QtGui.QColor(255, 255, 255, 220))
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)

        msg = (
            "Gaze overlay not calibrated.\n"
            "Type 'c' in the terminal to start AUTO calibration.\n"
            "Follow the red dot for ~10–20 seconds."
        )
        painter.drawText(rect.adjusted(12, 10, -12, -10),
                         QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, msg)

    def keyPressEvent(self, e):
        if e.key() in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            QtWidgets.QApplication.quit()


# -----------------------------
# Console hotkeys
# -----------------------------
def console_hotkey_loop(overlay: OverlayWindow):
    while True:
        try:
            ch = input().strip().lower()
        except EOFError:
            break

        if ch == "c":
            overlay.start_auto_calibration()
        elif ch == "r":
            overlay.reset_calibration()
        elif ch in ("q", "quit", "exit"):
            QtWidgets.QApplication.quit()
            break
        else:
            print("Commands: c (auto calibrate), r (reset), q (quit)")


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
    ap.add_argument("--debug-camera", action="store_true")
    ap.add_argument("--calib-points", type=int, default=12, help="9, 12, 16 are good choices.")
    ap.add_argument("--calib-settle-ms", type=int, default=450)
    ap.add_argument("--calib-collect-ms", type=int, default=950)
    ap.add_argument("--smoothing", type=float, default=0.22)
    return ap.parse_args()


def main():
    args = parse_args()

    worker = GazeWorker(
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        device_str=args.device,
        eye=args.eye,
        camera_index=args.camera,
        show_debug=args.debug_camera,
    )
    worker.start()

    app = QtWidgets.QApplication([])

    overlay = OverlayWindow(
        worker,
        calib_points=args.calib_points,
        settle_ms=args.calib_settle_ms,
        collect_ms=args.calib_collect_ms,
        smoothing=args.smoothing,
    )

    t = threading.Thread(target=console_hotkey_loop, args=(overlay,), daemon=True)
    t.start()

    try:
        app.exec_()
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
