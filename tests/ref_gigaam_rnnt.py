"""onnx-asr reference for GigaAM v3 E2E RNNT (text + word timestamps).

Usage: python tests/ref_gigaam_rnnt.py [wav] [out.json]
"""

import json
import sys

import numpy as np
import onnx_asr
import soundfile as sf

MODEL_DIR = "models/gigaam-v3-rnnt"


def main() -> None:
    wav = sys.argv[1] if len(sys.argv) > 1 else "models/tests_sample.wav"
    out = sys.argv[2] if len(sys.argv) > 2 else None

    wave, sample_rate = sf.read(wav, dtype="float32")
    assert sample_rate == 16_000, sample_rate
    if wave.ndim > 1:
        wave = wave.mean(axis=1)

    model = onnx_asr.load_model("gigaam-v3-e2e-rnnt", path=MODEL_DIR).with_timestamps()
    res = model.recognize(wave)

    doc = {
        "text": res.text,
        "tokens": list(res.tokens or []),
        "timestamps": [round(float(t), 3) for t in (res.timestamps or [])],
        "logprobs": [round(float(p), 4) for p in (res.logprobs or [])],
    }
    print("TEXT:", doc["text"])
    print(f"tokens: {len(doc['tokens'])}")
    for tok, ts in list(zip(doc["tokens"], doc["timestamps"]))[:15]:
        print(f"   {ts:7.2f}s  {tok!r}")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print("saved:", out)


if __name__ == "__main__":
    main()
