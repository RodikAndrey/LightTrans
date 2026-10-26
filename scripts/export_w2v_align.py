"""Export a WhisperX-style wav2vec2 forced-alignment model to ONNX.

Source: jonatasgrosman/wav2vec2-large-xlsr-53-russian (the default WhisperX
aligner for ru). Exports logits [B, T', C] for a waveform [B, T] (16 kHz f32;
normalization to zero mean / unit std is done at runtime in Rust, matching
Wav2Vec2FeatureExtractor(do_normalize=True)).

Outputs to models/wav2vec2-align-ru/:
  model.onnx       — fp16 graph (~0.6 GB)
  vocab.json       — {"<pad>": 0, ...} char -> id
  meta.json        — blank/sep/unk ids, frame duration, normalization flag

Also validates ONNX logits against the fp32 PyTorch model on a test wav.
"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC

SRC = "deps/w2v-ru-hf"
OUT = "models/wav2vec2-align-ru"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    model = Wav2Vec2ForCTC.from_pretrained(SRC).eval()
    vocab = json.load(open(os.path.join(SRC, "vocab.json"), encoding="utf-8"))

    wav, sr = sf.read(sys.argv[1] if len(sys.argv) > 1 else "tests/sample_ru_16k.wav", dtype="float32")
    assert sr == 16_000, sr
    x = torch.from_numpy((wav - wav.mean()) / np.sqrt(wav.var() + 1e-7))[None, :]

    with torch.inference_mode():
        ref = model(x).logits[0].numpy()  # [T', C] fp32

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, values):
            return self.m(values).logits

    half = Wrapper(model.half().cuda())
    x16 = x.half().cuda()
    with torch.inference_mode():
        ref16 = half(x16).float().cpu().numpy()[0]
    print(f"fp16 vs fp32 logits: max abs diff {np.abs(ref16 - ref).max():.4f}, "
          f"argmax agreement {(ref16.argmax(-1) == ref.argmax(-1)).mean() * 100:.2f}%")

    torch.onnx.export(
        half,
        (x16,),
        os.path.join(OUT, "model.onnx"),
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch", 1: "samples"}, "logits": {0: "batch", 1: "frames"}},
        opset_version=17,
    )

    so = ort.SessionOptions()
    sess = ort.InferenceSession(os.path.join(OUT, "model.onnx"), so, providers=["CUDAExecutionProvider"])
    (out,) = sess.run(["logits"], {"waveform": x16.cpu().numpy()})
    out = out[0].astype(np.float32)
    print(f"onnx vs fp32 torch: max abs diff {np.abs(out - ref).max():.4f}, "
          f"argmax agreement {(out.argmax(-1) == ref.argmax(-1)).mean() * 100:.2f}%")
    print("frames:", out.shape, "expected ~ samples/320:", len(wav) / 320)

    json.dump(vocab, open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(
        {
            "blank": vocab["<pad>"],
            "sep": vocab["|"],
            "unk": vocab["<unk>"],
            "frame_s": 0.02,
            "normalize": True,
            "language": "ru",
        },
        open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"),
    )
    print("saved", OUT)


if __name__ == "__main__":
    main()
