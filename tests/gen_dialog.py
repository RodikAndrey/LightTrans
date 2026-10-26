# Generates a two-speaker Russian dialog wav + ground truth (speaker per sentence)
# for diarization validation: alternating voices with short gaps.
import asyncio
import json
import sys

import edge_tts
import numpy as np
import soundfile as sf

SR = 16000
GAP = 0.35          # silence between turns, seconds
LEAD = 0.5          # leading silence

SENTENCES = [
    ("A", "Добрый день, меня зовут Андрей, я руководитель проекта."),
    ("B", "Здравствуйте Андрей, меня зовут Мария, я аналитик данных."),
    ("A", "Расскажите, пожалуйста, о результатах последнего исследования."),
    ("B", "Конечно, мы опросили более тысячи клиентов за три месяца."),
    ("A", "И какие основные выводы вы можете сделать?"),
    ("B", "Большинство пользователей довольны скоростью работы приложения."),
    ("A", "А что говорят про интерфейс и удобство навигации?"),
    ("B", "Здесь есть замечания, люди просят упростить главное меню."),
    ("A", "Понял, давайте обсудим это на следующей неделе с дизайнерами."),
    ("B", "Хорошо, я подготовлю презентацию с цифрами и примерами."),
    ("A", "Отлично, тогда жду вас в четверг в десять часов утра."),
    ("B", "Договорились, всего доброго, до свидания."),
]


async def synth(text: str, voice: str, path: str):
    for attempt in range(5):
        try:
            tts = edge_tts.Communicate(text, voice)
            await tts.save(path)
            return
        except Exception as e:
            print(f"retry {attempt}: {e}", file=sys.stderr)
            await asyncio.sleep(2)
    raise RuntimeError(f"tts failed: {text}")


async def main():
    out_wav = sys.argv[1] if len(sys.argv) > 1 else "tests/dialog_2spk.wav"
    out_gt = sys.argv[2] if len(sys.argv) > 2 else "tests/dialog_2spk_gt.json"
    voices = {
        "A": "ru-RU-DmitryNeural",
        "B": "ru-RU-SvetlanaNeural",
    }
    pieces = []
    gt = []
    cursor = LEAD
    for i, (spk, text) in enumerate(SENTENCES):
        tmp = f"tests/_dlg_{i}.mp3"
        await synth(text, voices[spk], tmp)
        import subprocess
        wav_tmp = tmp.replace(".mp3", ".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
             "-ar", str(SR), "-ac", "1", wav_tmp], check=True)
        data, sr = sf.read(wav_tmp, dtype="float32")
        assert sr == SR
        pieces.append((cursor, data))
        gt.append({"speaker": spk, "start": round(cursor, 3),
                   "end": round(cursor + len(data) / SR, 3), "text": text})
        cursor += len(data) / SR + GAP

    total = int((cursor) * SR) + SR
    audio = np.zeros(total, dtype=np.float32)
    for start, data in pieces:
        s = int(start * SR)
        audio[s:s + len(data)] += data

    sf.write(out_wav, audio, SR)
    with open(out_gt, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=1)
    print(f"wrote {out_wav} ({total / SR:.1f}s) and {out_gt}")


if __name__ == "__main__":
    asyncio.run(main())
