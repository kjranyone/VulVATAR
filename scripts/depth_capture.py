#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9,<3.13"
# dependencies = [
#     "pyrealsense2",
#     "numpy",
#     "PySide6-Essentials",  # QtCore/QtGui/QtWidgets only — skips the 161 MB addons
# ]
# ///
"""D435 capture utility (GUI) for the depth-camera pose rebuild.

A PySide6 desktop app for the Intel RealSense D435: live RGB + colorized
depth preview with clickable controls to record labelled `.db3` clips
(raw depth + colour streams, with intrinsics) that the offline PoC
replays via pyrealsense2. Pick a label, click Record, click Stop.

Runs on the machine with the camera attached (needs a display). It is a
dev tool, not part of the Rust pipeline.

    uv run --script scripts/depth_capture.py   # deps come from the PEP 723 block above

Recordings land in `diagnostics/depth/<label>_<timestamp>.db3`
(gitignored). The on-screen "depth valid" percentage and the centre-pixel
range are sanity checks: if the hands read 0 mm / show big holes at the
pose you care about, the capture is no good — reframe and re-record.
"""

from __future__ import annotations

import sys
import time
import threading
from datetime import datetime
from pathlib import Path

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit(
        "pyrealsense2 is not installed.\n"
        "  run via:  uv run --script scripts/depth_capture.py\n"
        "  (deps are declared inline via PEP 723 and provisioned by uv).\n"
        "Bare-python fallback: pip install pyrealsense2 numpy PySide6-Essentials"
    )

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

# --- capture config -------------------------------------------------------
DEPTH_W, DEPTH_H, DEPTH_FPS = 848, 480, 30
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
LABELS = ["setup", "wave", "palms_front", "namaste", "framing", "freeform"]

OUT_DIR = Path(__file__).resolve().parent.parent / "diagnostics" / "depth"


def _to_qimage_rgb(bgr: np.ndarray) -> QImage:
    """BGR uint8 HxWx3 (RealSense/colorizer order) -> owned RGB888 QImage."""
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    # .copy() so the QImage owns its pixels — the source bytes are freed
    # as soon as this returns (and it is handed to the GUI thread).
    return QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()


class CaptureThread(QThread):
    """Owns the RealSense pipeline; emits frames + stats to the GUI.

    The pipeline is single-thread-affine, so recording start/stop (which
    restarts the pipeline with/without a record config) and single-frame
    dumps are requested via thread-safe flags and executed here, inside
    the capture loop — never from the GUI thread.
    """

    frame_ready = Signal(QImage, QImage, dict)
    error = Signal(str, bool)  # message, fatal (True => close app)
    recording_changed = Signal(bool, str)  # is_recording, path
    dumped = Signal(str)                    # dump file stem

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._stop = False
        self._pending_record_start: str | None = None
        self._pending_record_stop = False
        self._pending_dump: str | None = None

    # --- GUI-thread request API (thread-safe) -----------------------------
    def start_recording(self, label: str) -> None:
        with self._lock:
            self._pending_record_start = label

    def stop_recording(self) -> None:
        with self._lock:
            self._pending_record_stop = True

    def request_dump(self, label: str) -> None:
        with self._lock:
            self._pending_dump = label

    def stop(self) -> None:
        self._stop = True

    # --- pipeline ---------------------------------------------------------
    @staticmethod
    def _start_pipeline(record_path: str | None = None) -> rs.pipeline:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
        if record_path is not None:
            cfg.enable_record_to_file(record_path)
        pipe.start(cfg)
        return pipe

    def run(self) -> None:
        ctx = rs.context()
        if len(ctx.query_devices()) == 0:
            self.error.emit("No RealSense device found. Plug in the D435 (USB3) and retry.", True)
            return

        colorizer = rs.colorizer()
        align = rs.align(rs.stream.color)
        try:
            pipe = self._start_pipeline()
        except Exception as exc:  # noqa: BLE001 — surface any camera-start failure
            self.error.emit(f"Failed to start camera: {exc}", True)
            return

        recording = False
        rec_path: str | None = None
        rec_start: float | None = None
        t_prev, fps = time.time(), 0.0

        try:
            while not self._stop:
                with self._lock:
                    start_lbl = self._pending_record_start
                    self._pending_record_start = None
                    stop_req = self._pending_record_stop
                    self._pending_record_stop = False
                    dump_lbl = self._pending_dump
                    self._pending_dump = None

                if start_lbl is not None and not recording:
                    rec_path = str(OUT_DIR / f"{start_lbl}_{datetime.now():%Y%m%d_%H%M%S}.db3")
                    try:
                        pipe.stop()
                        pipe = self._start_pipeline(rec_path)
                        recording, rec_start = True, time.time()
                        self.recording_changed.emit(True, rec_path)
                    except Exception as exc:  # noqa: BLE001
                        self.error.emit(f"record start failed: {exc}", False)
                elif stop_req and recording:
                    try:
                        pipe.stop()
                        pipe = self._start_pipeline()
                    except Exception as exc:  # noqa: BLE001
                        self.error.emit(f"record stop failed: {exc}", False)
                    recording = False
                    self.recording_changed.emit(False, rec_path or "")

                try:
                    frames = pipe.wait_for_frames(2000)
                except RuntimeError:
                    continue
                frames = align.process(frames)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if not depth or not color:
                    continue

                color_img = np.asanyarray(color.get_data())  # HxWx3 bgr8
                depth_raw = np.asanyarray(depth.get_data())   # uint16, mm
                depth_vis = np.asanyarray(colorizer.colorize(depth).get_data())

                if dump_lbl is not None:
                    stem = str(OUT_DIR / f"dump_{dump_lbl}_{datetime.now():%Y%m%d_%H%M%S}")
                    _to_qimage_rgb(color_img).save(stem + "_color.png")
                    np.save(stem + "_depth_mm.npy", depth_raw)
                    self.dumped.emit(stem)

                h, w = depth_raw.shape
                roi = depth_raw[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
                valid_pct = 100.0 * np.count_nonzero(roi) / roi.size
                center_mm = int(depth_raw[h // 2, w // 2])

                now = time.time()
                dt = now - t_prev
                t_prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                elapsed = (now - rec_start) if recording and rec_start else 0.0

                self.frame_ready.emit(
                    _to_qimage_rgb(color_img),
                    _to_qimage_rgb(depth_vis),
                    {
                        "valid_pct": valid_pct,
                        "center_mm": center_mm,
                        "fps": fps,
                        "recording": recording,
                        "elapsed": elapsed,
                        "rec_path": rec_path,
                    },
                )
        finally:
            try:
                pipe.stop()
            except Exception:  # noqa: BLE001
                pass


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VulVATAR — D435 depth capture")
        self.recorded: list[str] = []

        # --- live preview panes ------------------------------------------
        self.color_view = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.depth_view = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        for v in (self.color_view, self.depth_view):
            v.setMinimumSize(480, 360)
            v.setStyleSheet("background:#111;")

        color_col = QVBoxLayout()
        color_col.addWidget(QLabel("RGB"))
        color_col.addWidget(self.color_view)
        depth_col = QVBoxLayout()
        depth_col.addWidget(QLabel("Depth (colorized)"))
        depth_col.addWidget(self.depth_view)
        preview = QHBoxLayout()
        preview.addLayout(color_col)
        preview.addLayout(depth_col)

        # --- clip label selector -----------------------------------------
        self.label_group = QButtonGroup(self)
        self.radios: list[QRadioButton] = []
        label_box = QGroupBox("Clip label")
        label_col = QVBoxLayout()
        for i, name in enumerate(LABELS):
            rb = QRadioButton(name)
            if i == 0:
                rb.setChecked(True)
            self.label_group.addButton(rb, i)
            label_col.addWidget(rb)
            self.radios.append(rb)
        label_box.setLayout(label_col)

        # --- controls -----------------------------------------------------
        self.rec_btn = QPushButton("● Record")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setMinimumHeight(44)
        self.dump_btn = QPushButton("Dump single frame")
        self.quit_btn = QPushButton("Quit")

        self.stat_label = QLabel("starting camera…")
        self.stat_label.setWordWrap(True)
        self.rec_status = QLabel("idle")
        self.rec_status.setWordWrap(True)
        self.recorded_label = QLabel("(none yet)")
        self.recorded_label.setWordWrap(True)

        controls = QVBoxLayout()
        controls.addWidget(label_box)
        controls.addWidget(self.rec_btn)
        controls.addWidget(self.dump_btn)
        controls.addSpacing(8)
        controls.addWidget(self.stat_label)
        controls.addWidget(self.rec_status)
        controls.addSpacing(8)
        controls.addWidget(QLabel("Recorded clips:"))
        controls.addWidget(self.recorded_label)
        controls.addStretch(1)
        controls.addWidget(self.quit_btn)

        root = QHBoxLayout(self)
        root.addLayout(preview, 3)
        root.addLayout(controls, 1)

        # --- capture thread ----------------------------------------------
        self.cap = CaptureThread()
        self.cap.frame_ready.connect(self.on_frame)
        self.cap.error.connect(self.on_error)
        self.cap.recording_changed.connect(self.on_recording_changed)
        self.cap.dumped.connect(self.on_dumped)

        self.rec_btn.clicked.connect(self.on_rec_clicked)
        self.dump_btn.clicked.connect(self.on_dump_clicked)
        self.quit_btn.clicked.connect(self.close)

        self.cap.start()

    # --- helpers ----------------------------------------------------------
    def _current_label(self) -> str:
        return LABELS[self.label_group.checkedId()]

    # --- slots ------------------------------------------------------------
    def on_frame(self, color_img: QImage, depth_img: QImage, stats: dict) -> None:
        self.color_view.setPixmap(
            QPixmap.fromImage(color_img).scaled(
                self.color_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.depth_view.setPixmap(
            QPixmap.fromImage(depth_img).scaled(
                self.depth_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.stat_label.setText(
            f"depth valid (center): {stats['valid_pct']:.1f}%\n"
            f"center range: {stats['center_mm']} mm\n"
            f"{stats['fps']:.1f} fps"
        )
        if stats["recording"]:
            name = Path(stats["rec_path"]).name if stats["rec_path"] else ""
            self.rec_status.setText(f"● REC  {stats['elapsed']:.1f}s\n{name}")

    def on_rec_clicked(self) -> None:
        if self.rec_btn.isChecked():
            self.cap.start_recording(self._current_label())
        else:
            self.cap.stop_recording()

    def on_dump_clicked(self) -> None:
        self.cap.request_dump(self._current_label())

    def on_recording_changed(self, is_recording: bool, path: str) -> None:
        self.rec_btn.setChecked(is_recording)
        self.rec_btn.setText("■ Stop" if is_recording else "● Record")
        self.rec_btn.setStyleSheet(
            "background:#c0392b; color:white; font-weight:bold;" if is_recording else ""
        )
        for rb in self.radios:
            rb.setEnabled(not is_recording)
        self.dump_btn.setEnabled(not is_recording)
        if not is_recording:
            self.rec_status.setText("idle")
            if path:
                self.recorded.append(path)
                self.recorded_label.setText(
                    "\n".join(Path(p).name for p in self.recorded)
                )

    def on_dumped(self, stem: str) -> None:
        self.rec_status.setText(f"dumped {Path(stem).name}_color.png / _depth_mm.npy")

    def on_error(self, message: str, fatal: bool) -> None:
        QMessageBox.critical(self, "Capture error", message)
        if fatal:
            self.close()
        else:
            # transient (record/dump) failure — reset to idle, keep running
            self.rec_btn.setChecked(False)
            self.on_recording_changed(False, "")

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt override
        self.cap.stop()
        self.cap.wait(3000)
        event.accept()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1280, 680)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
