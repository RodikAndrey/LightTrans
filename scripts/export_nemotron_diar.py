# Export nvidia/Nemotron-3-Diarization core (encoder + heads) to ONNX for LightTrans.
#
# The exported graph covers ONE chunk step of the offline/streaming forward:
#   inputs:
#     cache_embeds : [C, 512]   arrival-order speaker cache embeds (C may be 0)
#     fifo_embeds  : [F, 512]   FIFO queue embeds (F may be 0)
#     chunk_mel    : [T, 128]   mel frames of this chunk (T % 8 == 0), includes right context
#     position_ids : [S] int64  S = C + F + T/8 (positions restart at every chunk)
#   outputs:
#     logits       : [S*8, 8]   per-frame speaker logits (10 ms frames), Rust slices the chunk part
#     chunk_embeds : [T/8, 512] embedder outputs, feed back via cache/fifo on the next step
#
# The speaker-cache (AOSC) bookkeeping itself is plain tensor shuffling and lives in Rust (src/diarize.rs).
import json
import math
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "F:/LightTrans/deps/nemotron-ref")
from hf_nemo_ref.configuration_nemotron3_diarization import Nemotron3DiarizationConfig
from hf_nemo_ref.modeling_nemotron3_diarization import (
    Nemotron3DiarizationForAudioFrameClassification,
)

MODEL_DIR = "F:/LightTrans/models/nemotron3-diarization"


class DiarStep(nn.Module):
    def __init__(self, m: Nemotron3DiarizationForAudioFrameClassification):
        super().__init__()
        self.m = m

    def forward(self, cache_embeds, fifo_embeds, chunk_mel, position_ids):
        chunk_embeds = self.m.model.audio_tower.embedder(chunk_mel[None])  # [1, T/8, 512]
        x = torch.cat([cache_embeds, fifo_embeds, chunk_embeds[0]], dim=0)  # [S, 512]
        hidden = self.m.model.audio_tower(
            inputs_embeds=x[None], position_ids=position_ids[None], attention_mask=None
        ).last_hidden_state  # [1, S, 512]
        h = self.m.model.proj(hidden[0])  # [S, 192]
        up = self.m.model.upsampler(h[None])  # [1, S*8, 192]
        logits = self.m.classifier(up)  # [1, S*8, 8]
        return logits[0], chunk_embeds[0]


def main():
    cfg = Nemotron3DiarizationConfig.from_json_file(f"{MODEL_DIR}/config.json")
    cfg._attn_implementation = "eager"
    m = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(MODEL_DIR, config=cfg)
    m.eval()

    wrapper = DiarStep(m)
    C, F, T = 264, 40, 2720  # representative shapes; all dynamic at runtime
    args = (
        torch.randn(C, 512),
        torch.randn(F, 512),
        torch.randn(T, 128),
        torch.arange(C + F + T // 8, dtype=torch.int64),
    )
    out_path = f"{MODEL_DIR}/diar_step.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            out_path,
            input_names=["cache_embeds", "fifo_embeds", "chunk_mel", "position_ids"],
            output_names=["logits", "chunk_embeds"],
            dynamic_axes={
                "cache_embeds": {0: "C"},
                "fifo_embeds": {0: "F"},
                "chunk_mel": {0: "T"},
                "position_ids": {0: "S"},
                "logits": {0: "S8"},
                "chunk_embeds": {0: "E"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print("exported:", out_path)

    # numeric check: eager wrapper vs ONNX on random shapes incl. empty cache/fifo
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for (c, f, t) in [(264, 40, 2720), (0, 0, 640), (7, 5, 64), (264, 12, 800)]:
        ca = torch.randn(c, 512)
        fi = torch.randn(f, 512)
        me = torch.randn(t, 128) * 3.0
        pos = torch.arange(c + f + t // 8, dtype=torch.int64)
        with torch.no_grad():
            ref_logits, ref_emb = wrapper(ca, fi, me, pos)
        o_logits, o_emb = sess.run(
            None,
            {
                "cache_embeds": ca.numpy(),
                "fifo_embeds": fi.numpy(),
                "chunk_mel": me.numpy(),
                "position_ids": pos.numpy(),
            },
        )
        dl = np.abs(o_logits - ref_logits.numpy()).max()
        de = np.abs(o_emb - ref_emb.numpy()).max()
        print(f"C={c} F={f} T={t}: max|dlogits|={dl:.2e} max|demb|={de:.2e}")
        assert dl < 2e-4 and de < 2e-4, "numeric mismatch"

    # librosa mel filterbank (slaney scale + slaney norm) for the Rust side
    import librosa

    fb = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000.0, norm="slaney")
    assert fb.shape == (128, 257), fb.shape
    fb.astype(np.float32).tofile(f"{MODEL_DIR}/mel_filters.f32")

    meta = {
        "sampling_rate": 16000,
        "n_fft": 512,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": 128,
        "preemphasis": 0.97,
        "log_zero_guard": 2**-24,
        "num_speakers": 8,
        "subsampling_factor": 8,
        # offline chunking config (encoder frames of 80 ms)
        "chunk_length": cfg.chunk_length,
        "chunk_right_context": cfg.chunk_right_context,
        "fifo_length": cfg.fifo_length,
        "speaker_cache_length": cfg.streaming_config.speaker_cache_length,
        "speaker_cache_update_period": cfg.speaker_cache_update_period,
        "speaker_cache_silence_frames_per_speaker": cfg.streaming_config.speaker_cache_silence_frames_per_speaker,
        "prediction_score_threshold": cfg.streaming_config.prediction_score_threshold,
        "latest_frames_score_boost": cfg.streaming_config.latest_frames_score_boost,
        "min_positive_scores_rate": cfg.streaming_config.min_positive_scores_rate,
        "strong_boost_rate": cfg.streaming_config.strong_boost_rate,
        "weak_boost_rate": cfg.streaming_config.weak_boost_rate,
    }
    with open(f"{MODEL_DIR}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print("mel_filters.f32 + meta.json written")
    print("ALL OK")


if __name__ == "__main__":
    main()
