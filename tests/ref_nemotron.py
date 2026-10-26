# Reference dump for Nemotron-3-Diarization: HF mel features + offline logits + segments.
# Used to validate the Rust implementation numerically (mel comparison, logits comparison).
import json
import sys

sys.path.insert(0, "F:/LightTrans/deps/nemotron-ref")

import numpy as np
import soundfile as sf
import torch

from transformers import AutoProcessor

MODEL_DIR = "F:/LightTrans/models/nemotron3-diarization"


def main():
    wav = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else wav.rsplit(".", 1)[0] + "_ref"

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == 16000, sr
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(-1)

    inputs = processor(audio, sampling_rate=16000)
    mel = inputs["input_features"][0].numpy()  # [T, 128]
    mask = inputs["attention_mask"][0].numpy().astype(np.int8)  # [T]
    print(f"audio {len(audio)/16000:.2f}s, mel frames {mel.shape[0]}, valid {int(mask.sum())}")

    np.save(f"{out_prefix}_mel.npy", mel)
    np.save(f"{out_prefix}_mask.npy", mask)

    from hf_nemo_ref.modeling_nemotron3_diarization import (
        Nemotron3DiarizationForAudioFrameClassification,
    )
    from hf_nemo_ref.configuration_nemotron3_diarization import Nemotron3DiarizationConfig

    cfg = Nemotron3DiarizationConfig.from_json_file(f"{MODEL_DIR}/config.json")
    model = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(MODEL_DIR, config=cfg)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits.numpy()  # [1, T, 8]
    np.save(f"{out_prefix}_logits.npy", logits[0])

    # replicate Nemotron3DiarizationProcessor.extract_speaker_dict (not available in transformers 5.17)
    frame_duration = 160 / 16000
    active = (torch.tensor(logits[0]).sigmoid() > 0.5)
    mask_t = torch.tensor(mask.astype(bool))
    active = active & mask_t[..., None]
    segments = []
    for speaker in range(active.shape[1]):
        a = active[:, speaker].int()
        changes = torch.diff(torch.cat([torch.zeros(1, dtype=torch.int), a, torch.zeros(1, dtype=torch.int)]))
        starts = (changes == 1).nonzero()[:, 0].tolist()
        ends = (changes == -1).nonzero()[:, 0].tolist()
        segments.extend({"Start": round(s * frame_duration, 2), "End": round(e * frame_duration, 2),
                         "Speaker": speaker} for s, e in zip(starts, ends))
    segments.sort(key=lambda s: (s["Start"], s["Speaker"]))
    with open(f"{out_prefix}_segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=1)
    print(f"logits {logits.shape} -> {out_prefix}_logits.npy")
    print(f"segments: {len(segments)}")
    for s in segments[:12]:
        print(f"  spk{s['Speaker']}: {s['Start']:.2f} - {s['End']:.2f}")


if __name__ == "__main__":
    main()
