#!/usr/bin/env python3
"""Train a small viseme classification model and export to ONNX.

This script generates synthetic training data from formant patterns
of the five Japanese vowels (a, i, u, e, o) + silence, trains a tiny
1D-CNN classifier on log-mel spectrogram features, and exports the
result to models/viseme_nn.onnx.

Requirements:
    pip install torch torchaudio onnx numpy

Usage:
    python scripts/train_viseme_model.py [--epochs 200] [--output models/viseme_nn.onnx]

The generated model accepts input of shape [1, 40, N] (batch, mel_bins, time_frames)
and outputs [1, 6] logits for classes: aa, ih, ou, ee, oh, silent.
"""

import argparse
import os
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)


# ── Model architecture ────────────────────────────────────────────────

class VisemeNet(nn.Module):
    """Tiny 1D-CNN for viseme classification from log-mel spectrograms.

    Input:  [B, 40, T]  (mel bins, time frames; T is variable but typically 10)
    Output: [B, 6]      (logits for aa, ih, ou, ee, oh, silent)
    """

    def __init__(self, n_mels=40, n_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: [B, n_mels, T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # [B, 64]
        return self.fc(x)  # [B, 6]


# ── Synthetic data generation ─────────────────────────────────────────

# Approximate first two formant frequencies (Hz) for Japanese vowels.
# These are used to generate synthetic mel-like patterns.
VOWEL_FORMANTS = {
    "aa": (800, 1200),   # /a/ - open
    "ih": (300, 2300),   # /i/ - front close
    "ou": (350, 800),    # /u/ - back close
    "ee": (500, 1800),   # /e/ - front mid
    "oh": (500, 1000),   # /o/ - back mid
}
CLASS_NAMES = ["aa", "ih", "ou", "ee", "oh", "silent"]
N_CLASSES = len(CLASS_NAMES)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def generate_mel_pattern(class_idx, n_mels=40, n_frames=10, sample_rate=16000,
                         n_fft=512, fmin=60.0, fmax=7600.0):
    """Generate a synthetic log-mel spectrogram for a given class."""
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_centers = np.linspace(mel_min, mel_max, n_mels)

    if class_idx == 5:  # silent
        pattern = np.random.randn(n_mels, n_frames) * 0.1 - 5.0
        return pattern.astype(np.float32)

    name = CLASS_NAMES[class_idx]
    f1, f2 = VOWEL_FORMANTS[name]

    # Add random variation to formants.
    f1 += np.random.randn() * 50
    f2 += np.random.randn() * 80

    mel_f1 = hz_to_mel(f1)
    mel_f2 = hz_to_mel(f2)

    pattern = np.zeros((n_mels, n_frames), dtype=np.float32)
    for m in range(n_mels):
        mc = mel_centers[m]
        # Gaussian bumps at formant frequencies.
        energy = (np.exp(-0.5 * ((mc - mel_f1) / 60.0) ** 2) * 3.0 +
                  np.exp(-0.5 * ((mc - mel_f2) / 80.0) ** 2) * 2.0)
        # Add fundamental and noise.
        energy += 0.3
        pattern[m, :] = np.log(energy + 1e-6) + np.random.randn(n_frames) * 0.15

    # Random volume scaling.
    volume = np.random.uniform(0.5, 2.0)
    pattern += np.log(volume)

    return pattern


def generate_dataset(n_per_class=2000, n_mels=40, n_frames=10):
    """Generate a synthetic dataset of mel spectrograms."""
    X = []
    y = []
    for cls in range(N_CLASSES):
        for _ in range(n_per_class):
            mel = generate_mel_pattern(cls, n_mels=n_mels, n_frames=n_frames)
            X.append(mel)
            y.append(cls)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    # Shuffle.
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


# ── Training ──────────────────────────────────────────────────────────

def train(model, X_train, y_train, epochs=200, batch_size=128, lr=0.002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)

    n = len(y_train)
    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        correct = 0
        batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_t[idx]
            yb = y_t[idx]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            batches += 1

        if epoch % 20 == 0 or epoch == 1:
            acc = correct / n * 100
            print(f"  epoch {epoch:4d}/{epochs}  loss={total_loss/batches:.4f}  acc={acc:.1f}%")

    model.eval()
    return model


# ── ONNX export ───────────────────────────────────────────────────────

def export_onnx(model, output_path, n_mels=40, n_frames=10):
    model.eval()
    dummy = torch.randn(1, n_mels, n_frames)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["mel_input"],
        output_names=["logits"],
        dynamic_axes={
            "mel_input": {2: "n_frames"},
            "logits": {},
        },
        opset_version=13,
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  exported: {output_path} ({size_kb:.0f} KB)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train viseme classification model")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", type=str, default="models/viseme_nn.onnx")
    parser.add_argument("--n-per-class", type=int, default=3000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("Generating synthetic training data...")
    X, y = generate_dataset(n_per_class=args.n_per_class)
    print(f"  dataset: {X.shape[0]} samples, {N_CLASSES} classes")

    print("Training VisemeNet...")
    model = VisemeNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {total_params:,}")
    model = train(model, X, y, epochs=args.epochs)

    print("Exporting to ONNX...")
    export_onnx(model, args.output)

    # Quick validation.
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X[:1000])
        logits = model(X_t)
        preds = logits.argmax(1).numpy()
        acc = (preds == y[:1000]).mean() * 100
    print(f"  validation accuracy: {acc:.1f}%")
    print("Done.")


if __name__ == "__main__":
    main()
