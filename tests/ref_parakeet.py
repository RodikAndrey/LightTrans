# Reference Parakeet TDT v3 transcription via onnx-asr (local ONNX export).
# Usage: python tests/ref_parakeet.py <wav> [<wav> ...]
import sys

import numpy as np
import soundfile as sf

import onnx_asr

MODEL_DIR = "models/parakeet-v3"


def main():
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", path=MODEL_DIR).with_timestamps()
    for wav in sys.argv[1:]:
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(-1)
        assert sr == 16000, sr
        res = model.recognize(audio)
        # TimestampedResult: text, timestamps (s), tokens, logprobs
        print(f"=== {wav} ===")
        print("text:", res.text)
        if res.timestamps:
            toks = list(zip(res.timestamps, res.tokens))
            print(f"tokens: {len(toks)}, first 12:", [(round(t, 2), n) for t, n in toks[:12]])
        import json
        out = wav.rsplit(".", 1)[0] + "_parakeet_ref.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "text": res.text,
                "timestamps": list(res.timestamps or []),
                "tokens": list(res.tokens or []),
                "logprobs": list(res.logprobs or []),
            }, f, ensure_ascii=False)
        print("saved:", out)


if __name__ == "__main__":
    main()
