"""Generate word-timestamp ground-truth test audio.

Each word is synthesized separately (edge-tts), silence-trimmed, then placed at
an exactly known offset on a silent timeline via ffmpeg. Ground truth JSON is
written next to the wav.

Usage: python gen_word_timing_test.py <lang> <out_prefix>
  lang: ru | en
"""

import json
import subprocess
import sys
from pathlib import Path

import edge_tts
import soundfile as sf

WORDS = {
    "ru": ["телефон", "квартира", "мелодия", "гитара", "обед", "сатира",
           "экран", "знание", "погода", "машина", "урок", "восток"],
    "en": ["apple", "banana", "computer", "elephant", "guitar",
           "ocean", "planet", "rocket", "window", "forest", "market", "silver"],
}
VOICE = {"ru": "ru-RU-DmitryNeural", "en": "en-US-GuyNeural"}
GAP = 1.0       # seconds between word starts (isolated words, no overlap)
LEAD = 0.7      # silence before the first word
PAD = 0.05      # silence padding left at each word edge by the trimmer


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def trim(path: Path, out: Path) -> float:
    """Silence-trim both ends of a wav, leaving PAD seconds at each edge."""
    filt = (
        f"silenceremove=start_periods=1:start_silence={PAD}:start_threshold=-45dB,"
        "areverse,"
        f"silenceremove=start_periods=1:start_silence={PAD}:start_threshold=-45dB,"
        "areverse"
    )
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-af", filt, "-ar", "16000", "-ac", "1", str(out)])
    data, sr = sf.read(out)
    return len(data) / sr


async def synth(text: str, voice: str, out: Path) -> None:
    import asyncio

    for attempt in range(5):
        try:
            c = edge_tts.Communicate(text, voice)
            await c.save(str(out))
            return
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  retry {attempt + 1} for '{text}': {e}")
            await asyncio.sleep(2)


def main() -> None:
    lang, prefix = sys.argv[1], sys.argv[2]
    tmp = Path("tmp_words")
    tmp.mkdir(exist_ok=True)

    clips = []
    import asyncio

    for i, word in enumerate(WORDS[lang]):
        mp3 = tmp / f"w{i}.mp3"
        raw = tmp / f"w{i}_raw.wav"
        trimmed = tmp / f"w{i}.wav"
        asyncio.run(synth(word, VOICE[lang], mp3))
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ar", "16000",
             "-ac", "1", str(raw)])
        dur = trim(raw, trimmed)
        clips.append((word, trimmed, dur))
        print(f"  {word}: {dur:.3f}s")

    # exact placement
    offsets = [LEAD + i * GAP for i in range(len(clips))]
    total = offsets[-1] + max(d for _, _, d in clips) + 0.5

    inputs: list[str] = []
    filt: list[str] = []
    mix = ""
    for i, (_, wav, _) in enumerate(clips):
        inputs += ["-i", str(wav)]
        delay_ms = int(round(offsets[i] * 1000))
        filt.append(f"[{i}:a]adelay={delay_ms}:all=1[a{i}]")
        mix += f"[a{i}]"
    filt.append(f"{mix}amix=inputs={len(clips)}:normalize=0:duration=longest,"
                f"apad=whole_dur={total:.3f}[out]")

    out_wav = Path(f"{prefix}.wav")
    run(["ffmpeg", "-y", "-loglevel", "error", *inputs,
         "-filter_complex", ";".join(filt), "-map", "[out]",
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_f32le", str(out_wav)])

    gt = [{"word": w, "start": offsets[i], "end": offsets[i] + d}
          for i, (w, _, d) in enumerate(clips)]
    gt_path = Path(f"{prefix}_gt.json")
    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out_wav} ({total:.2f}s) and {gt_path}")


if __name__ == "__main__":
    main()
