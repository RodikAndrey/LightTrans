# Validates the Rust diarization against the HF reference and ground truth:
#  1. mel features: Rust (raw f32 dump) vs HF processor (npy)
#  2. per-frame speaker probabilities: Rust vs HF sigmoid(logits) on valid frames
#  3. word -> speaker assignment vs the dialog ground truth
import json
import sys

import numpy as np


def read_f32(path):
    return np.fromfile(path, dtype="<f4")


def main():
    ok = True

    # --- 1. mel ---
    ref_mel = np.load("tests/dialog_2spk_ref_mel.npy")  # [T_ref, 128]
    mask = np.load("tests/dialog_2spk_ref_mask.npy").astype(bool)
    rust_mel = read_f32("out/dump/diar_debug_mel.f32").reshape(-1, 128)
    n = min(len(ref_mel), len(rust_mel))
    d = np.abs(rust_mel[:n] - ref_mel[:n])
    print(f"mel: rust {rust_mel.shape} vs ref {ref_mel.shape}, "
          f"max|d| = {d.max():.4f}, mean|d| = {d.mean():.5f}")
    # mel values are log-scale (typically [-1, 20]); 0.05 abs tolerance is ~sub-frame
    if d.max() > 0.05:
        print("  MEL MISMATCH")
        ok = False

    # --- 2. probabilities ---
    ref_logits = np.load("tests/dialog_2spk_ref_logits.npy")  # [T_ref, 8]
    ref_probs = 1.0 / (1.0 + np.exp(-ref_logits))
    rust_probs = read_f32("out/dump/diar_debug_probs.f32").reshape(-1, 8)
    n = min(int(mask.sum()), len(rust_probs))
    d = np.abs(rust_probs[:n] - ref_probs[:n])
    print(f"probs: rust {rust_probs.shape} vs ref valid {int(mask.sum())}, "
          f"max|d| = {d.max():.4f}, mean|d| = {d.mean():.6f}")
    # end-to-end (mel + chunked cache) tolerance; speaker flips happen near 0.5
    argmax_rust = rust_probs[:n].argmax(1)
    argmax_ref = ref_probs[:n].argmax(1)
    agree = (argmax_rust == argmax_ref).mean()
    print(f"argmax speaker agreement: {agree*100:.2f}%")
    if d.max() > 0.1 or agree < 0.99:
        print("  PROBS MISMATCH")
        ok = False

    # --- 3. word speakers vs ground truth ---
    out = json.load(open("out/dlg/dialog_2spk.json", encoding="utf-8"))
    gt = json.load(open("tests/dialog_2spk_gt.json", encoding="utf-8"))
    # speaker index from label; map model labels -> gt speakers by best overlap
    words = []
    for seg in out["segments"]:
        for w in seg["words"]:
            words.append(w)
    print(f"words: {len(words)}")
    # ground-truth speaker per word time
    def gt_speaker(t):
        for g in gt:
            if g["start"] - 0.05 <= t <= g["end"] + 0.05:
                return g["speaker"]
        return None

    pairs = {}
    unmatched = 0
    for w in words:
        g = gt_speaker(w["start"])
        lab = w.get("speaker")
        if g is None or lab is None:
            unmatched += 1
            continue
        pairs.setdefault((lab, g), 0)
        pairs[(lab, g)] += 1
    print(f"label->gt pairs: {pairs}, unmatched words: {unmatched}")
    # best mapping label -> gt
    mapping = {}
    for lab in {lab for lab, _ in pairs}:
        mapping[lab] = max(gt for (l, gt), c in pairs.items() if l == lab)
    correct = sum(c for (lab, g), c in pairs.items() if mapping[lab] == g)
    total = sum(pairs.values())
    acc = correct / max(total, 1)
    print(f"word speaker accuracy: {correct}/{total} = {acc*100:.2f}%")
    if acc < 0.95:
        print("  SPEAKER ACCURACY LOW")
        ok = False

    print("CHECK:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
