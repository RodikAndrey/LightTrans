# Raw reference: nemo128.onnx -> encoder -> greedy TDT decode, printing
# (token, frame, time) so the Rust implementation can be compared step by step.
import sys

import numpy as np
import onnxruntime as rt
import soundfile as sf

D = "models/parakeet-v3"

vocab = {}
for line in open(f"{D}/vocab.txt", encoding="utf-8"):
    tok, i = line.rstrip("\n").rsplit(" ", 1)
    vocab[int(i)] = tok
vocab_size = len(vocab)
blank = next(i for i, t in vocab.items() if t == "<blk>")

sess_opt = rt.SessionOptions()
mel = rt.InferenceSession(f"{D}/nemo128.onnx", sess_opt, providers=["CPUExecutionProvider"])
enc = rt.InferenceSession(f"{D}/encoder-model.onnx", sess_opt, providers=["CPUExecutionProvider"])
dec = rt.InferenceSession(f"{D}/decoder_joint-model.onnx", sess_opt, providers=["CPUExecutionProvider"])

wav = sys.argv[1]
audio, sr = sf.read(wav, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(-1)
wave = audio[None, :].astype(np.float32)

features, flens = mel.run(["features", "features_lens"],
                          {"waveforms": wave, "waveforms_lens": np.array([len(audio)], dtype=np.int64)})
print(f"mel: features {features.shape}, lens {flens}")

enc_out, enc_lens = enc.run(["outputs", "encoded_lengths"],
                            {"audio_signal": features[:, :, : flens[0]], "length": flens})
print(f"encoder: out {enc_out.shape}, lens {enc_lens}")

state_shapes = {x.name: x.shape for x in dec.get_inputs()}
st1 = np.zeros(shape=(state_shapes["input_states_1"][0], 1, state_shapes["input_states_1"][2]), dtype=np.float32)
st2 = np.zeros(shape=(state_shapes["input_states_2"][0], 1, state_shapes["input_states_2"][2]), dtype=np.float32)

T = int(enc_lens[0])
prev = blank
t = 0
emitted = 0
tokens = []
while t < T:
    outputs, _, ns1, ns2 = dec.run(
        ["outputs", "prednet_lengths", "output_states_1", "output_states_2"],
        {
            "encoder_outputs": enc_out[:, :, t][:, :, None],
            "targets": np.array([[prev]], dtype=np.int32),
            "target_length": np.array([1], dtype=np.int32),
            "input_states_1": st1,
            "input_states_2": st2,
        },
    )
    logits = outputs.reshape(-1)
    tok_logits, dur_logits = logits[:vocab_size], logits[vocab_size:]
    token = int(np.argmax(tok_logits))
    step = int(np.argmax(dur_logits))
    if token != blank:
        st1, st2 = ns1, ns2
        prev = token
        tokens.append((token, t))
        emitted += 1
    if step > 0:
        t += step
        emitted = 0
    elif token == blank or emitted == 10:
        t += 1
        emitted = 0

print(f"emitted tokens: {len(tokens)}")
for tok_id, fr in tokens[:16]:
    print(f"  t={fr:3d} ({fr*0.08:.2f}s) {vocab[tok_id]!r}")
text = "".join(vocab[i] for i, _ in tokens).replace("▁", " ").strip()
print("text:", text[:150])
