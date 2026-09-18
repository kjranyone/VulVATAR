#!/usr/bin/env python
"""Face landmark sidecar — runs the ready-made RTMPose-Face-WFLW LiteRT
model (`models/rtm_face_fp16.tflite`) for the Rust runtime.

Protocol on stdin/stdout (length-prefixed, little-endian u32):
  -> 4 bytes payload length, then a 256x256 RGB frame (raw bytes, row
     major) of the FACE CROP,
  <- 4 bytes payload length, then 98*2 float32 little-endian landmark
     coordinates normalised to [0, 1] of the crop (WFLW98 topology,
     SimCC argmax decode, no sub-cell refinement).

"quit" (raw, no length prefix) terminates. The model is loaded once at
startup; inference is ~30-60 ms/frame on CPU (XNNPACK).

Install: pip install ai-edge-litert
"""
import os
import struct
import sys

import numpy as np

# resolve models/ relative to the script (scripts/../models) unless
# overridden — mirrors how the app locates its ONNX models.
MODEL = os.environ.get(
    "VULVATAR_FACE98_TFLITE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "models", "rtm_face_fp16.tflite"))
MODEL = os.path.normpath(MODEL)

from ai_edge_litert.interpreter import Interpreter  # noqa: E402

interp = Interpreter(model_path=MODEL, num_threads=max(2, (os.cpu_count() or 4) - 2))
interp.allocate_tensors()
inp = interp.get_input_details()[0]
ox, oy = interp.get_output_details()[0], interp.get_output_details()[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def landmarks(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    x = ((c - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    sx = interp.get_tensor(ox["index"])[0]
    sy = interp.get_tensor(oy["index"])[0]
    pts = np.stack([sx.argmax(1) / 512.0, sy.argmax(1) / 512.0], 1)
    return pts.astype(np.float32).reshape(-1)


def read_exact(n):
    buf = sys.stdin.buffer
    data = b""
    while len(data) < n:
        chunk = buf.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def main():
    out = sys.stdout.buffer
    while True:
        hdr = read_exact(4)
        if hdr is None:
            break
        (n,) = struct.unpack("<I", hdr)
        if n > 4 * 1024 * 1024:
            break
        payload = read_exact(n)
        if payload is None:
            break
        if payload == b"quit":
            break
        rgb = np.frombuffer(payload, np.uint8).reshape(256, 256, 3)
        pts = landmarks(rgb)
        out.write(struct.pack("<I", pts.nbytes))
        out.write(pts.tobytes())
        out.flush()


if __name__ == "__main__":
    main()
