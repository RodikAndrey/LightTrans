# lighttrans

Быстрая транскрибация аудио и видео на GPU (NVIDIA CUDA), повторяющая пайплайн
[WhisperX](https://github.com/m-bain/whisperx): декодирование любых ffmpeg-форматов →
VAD (Silero) → нарезка на чанки → параллельное распознавание → пословные таймстампы →
вывод в json/srt/vtt/tsv/txt.

Написан на Rust. Модель по умолчанию — **GigaAM v3 E2E CTC**. Поддерживает четыре модели:

| Модель | Язык | Бэкенд | Особенности |
|---|---|---|---|
| **GigaAM v3 E2E CTC** (Sber) | ru | ONNX Runtime + CUDA EP | пунктуация и ITN «из коробки», ~240M параметров, очень быстрая |
| **GigaAM v3 E2E RNN-T** (Sber) | ru | ONNX Runtime + CUDA EP | тот же энкодер, трансьюзер-голова (LSTM + joint): иногда точнее на переспросах/шуме |
| **Whisper Large v3 Turbo** | 99 языков | whisper.cpp (CUDA) | мультиязычность, `translate`, авторазличение языка |
| **Parakeet TDT 0.6B v3** (NVIDIA) | 25 европейских языков (вкл. ru) | ONNX Runtime + CUDA EP | TDT-трансьюзер: таймстампы из длительностей токенов, Robust к шуму |

## Возможности

- Вход: **любой** аудио/видео формат, который читает ffmpeg (mp4, mkv, mp3, wav, flac, ogg, webm, …) — через библиотеки libavformat/libavcodec/libswresample, без внешнего бинарника.
- GPU: CUDA (nvcc-сборка whisper.cpp + ONNX Runtime CUDA EP). Автоматический fallback на CPU.
- **Параллелизм** (как в WhisperX, только потоками): стриминг-конвейер `декод → VAD → нарезка → N GPU-воркеров → слияние`.
  Число воркеров определяется автоматически по свободной VRAM (NVML), можно задать `--workers`. Декодирование и VAD
  работают в отдельном потоке и перекрываются с инференсом.
- VAD: Silero v5, гистерезис onset/offset 0.5/0.363 и min_duration_on/off 0.1 c — как в WhisperX (pyannote-binarize); батчами по 64 окна.
- Таймстампы: пословные (GigaAM — из CTC-фреймов, whisper — из token timestamps) + score.
- Форматы вывода: `json` (схема WhisperX: segments + word_segments), `srt`, `vtt`, `tsv`, `txt`. Несколько входных файлов за запуск.
- **Выравнивание слов на выбор** (whisper-turbo): `--dtw` — DTW-таймстампы whisper.cpp
  (точные, но ~2.2× медленнее), `--w2v` — быстрый whisper (flash attention) + форс-выравнивание
  CTC-моделью GigaAM в отдельной фазе на освободившейся GPU (та же схема, что в WhisperX с
  wav2vec2; суммарно в 1.5× быстрее WhisperX), без ключей — без выравнивания
  (сегментные t0/t1, максимально быстро).
- **Сводная статистика этапов** (печатается в конце каждого запуска): когда какой этап
  стартовал, сколько длился, сколько чанков обработано, аудио-секунды и общий RTF.
- **Диаризация спикеров** (`--diarize`): NVIDIA Nemotron-3-Diarization (ONNX, до 8 спикеров,
  перекрывающаяся речь, спикеры нумеруются по времени первого появления). Спикер пишется
  в каждое слово и сегмент (json) и префиксом `[SPEAKER_XX]` в srt/vtt. см. раздел «Диаризация».
- **Многоканальный диалог** (`--split-channels`, `--turns`): каждый канал входа — отдельный
  спикер (раздельные VAD/чанкер, метки `SPEAKER_0N` и `channel` в словах); dual-mono-источники
  (корреляция каналов > 0.98) автоматически схлопываются в один канал. `--turns` строит реплики
  диалога: пауза ≥ `--turn-gap` режет реплику, реплики упорядочены по времени.
- **Подкоманды**: `lighttrans vad файл -o out.json` — интервалы речи Silero без распознавания
  (json); `lighttrans bench файл …` — полный прогон с машиночитаемой сводкой этапов и RTF
  (json в stdout, таблица — в stderr).
- **HTTP-сервис** `lighttrans-server` (порт из gigaam-v3-runtime): batch-`POST /v1/transcribe`
  (GigaAM v3 CTC/RNN-T), метрики Prometheus, `/health` `/readyz`, graceful shutdown,
  верификация моделей по манифестам и CUDA-назначений по отпечаткам; провайдер выбирается
  явно (`--ep cpu|cuda`), fallback запрещён. Подробности и статус фаз — [PORTING.md](PORTING.md).
- **WebSocket-стриминг** `GET /v1/stream` (MVP: mono, `model=ctc`, `emit=words`):
  живое аудио кадрами (pcm16/f32/alaw/ulaw, произвольная частота 8000..192000),
  события ревизий `words`/`stable`/`final` с пословными таймстампами; инкрементальный
  log-mel и прогоны энкодера по 30-секундному окну, сшивка швов окон, endpointing
  по тишине CTC. Клиент завершает `{"type":"end"}`; кадры > 16 МиБ — close 1009.

## Точность пословных таймстампов (проверено)

Валидация на синтетическом эталоне (TTS-слова на точно известных позициях, `tests/`):
и на перекрёстной сверке двух независимых движков (CTC-фреймы GigaAM vs DTW whisper.cpp)
на реальной речи — 452 совпадённых слова, расхождение стартов медианно ~100 мс.

| Модель | Δстарт (медиана) | Δконец (медиана) |
|---|---|---|
| GigaAM v3 (CTC-фреймы, 40 мс/фрейм) | +84 мс* | −87 мс |
| Whisper Turbo ru (DTW) | +18 мс | +51 мс |
| Whisper Turbo en (DTW) | +138 мс | +93 мс |

\* в эталон входит 50 мс тишины в начале TTS-клипа — фактическая ошибка ~30 мс.
Слово не начинается раньше конца предыдущего; паузы не растягивают слова
(пунктуация не влияет на тайминги); сегменты-галлюцинации на тишине отбрасываются.

## Диаризация

Опция `--diarize` включает разделение спикеров моделью
[nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
(Streaming-Sortformer: 31-слойный Transformer + RoPE, 100M параметров, выход — вероятности
8 спикеров на каждые 10 мс аудио). Ядро модели экспортировано в ONNX
(`scripts/export_nemotron_diar.py`), логика Arrival-Order Speaker Cache (FIFO, компрессия
кэша, score-бустинг) портирована на Rust 1:1 из HF transformers.

```bash
scripts/run.sh interview.mp4 -m whisper-turbo --diarize -f json,srt
```

- json: `words[].speaker`, `segments[].speaker` (метки `SPEAKER_00`, `SPEAKER_01`, … по порядку появления);
- srt/vtt: `[SPEAKER_00] текст`;
- привязка спикера к слову: argmax суммы вероятностей по фреймам внутри слова.

Модель кладётся в `models/nemotron3-diarization/` (см. «Модели»). Скорость: RTF ~0.007
(5-минутный файл — ~2 с), работает параллельно с транскрипцией.

**Проверено** (`tests/check_diarization.py`, сравнение с эталоном HF transformers на
синтетическом 2-спикерном диалоге):
- mel-признаки: макс. расхождение с HF 0.0005 (log-шкала);
- вероятности фреймов: макс. 0.003, согласованность argmax-спикера 99.98 %;
- **привязка спикеров к словам: 92/92 = 100 %**.

## Производительность

NVIDIA RTX 3060 Laptop 6 GB, Windows, GPU-обработка:

| Тест | Аудио | Модель | Время (обработка) | RTF |
|---|---|---|---|---|
| sample 26 c (wav) | 26 с | GigaAM v3 | 0.15 с | 0.006 |
| sample 26 c (wav) | 26 с | GigaAM v3 RNN-T | 0.25 с | 0.010 |
| клип 5 мин (mp4) | 300 с | GigaAM v3, 3 воркера | ~3 с | ~0.010 |
| **видео 52.6 мин** (mp4) | 3156 с | GigaAM v3, 3 воркера | **~26 с** (вкл. загрузку 10 с) | **~0.008** |
| клип 5 мин (mp4) | 300 с | Whisper Turbo, 1 воркер | ~16 с | ~0.055 |
| видео 93 мин (mp4) | 5589 с | Whisper Turbo **--w2v**, 2 воркера | **~147 с** | **~0.026** |
| видео 93 мин (mp4) | 5589 с | Whisper Turbo --dtw, 2 воркера | ~306 с | ~0.055 |
| видео 93 мин (mp4) | 5589 с | WhisperX (faster-whisper, тот же файл) | ~215 с | ~0.038 |
| клип 5 мин (mp4) | 300 с | Parakeet TDT v3, 1 воркер | ~8 с (без декода/загрузки) | ~0.027 |

Декодирование mp4/aac: RTF 0.002–0.006 (52 мин ≈ 15–20 с, многопоточно).
Одноразовая загрузка моделей при старте: GigaAM ~10 с (3 сессии), Whisper ~2 с.

Масштабирование воркеров (клип 5 мин, 6 GB VRAM): whisper 1→2 воркера почти
линейно (окно 11.5→6.1 с), но 3-й воркер — провал (VRAM 5.9/6.0 ГБ, CUDA
выгружает в shared memory, чанк в ~5× медленнее); GigaAM CTC/RNN-T масштабируются
до 3 воркеров (~2.5×), Parakeet насыщает GPU одним воркером (2-й не даёт прироста).
`--workers auto` уже выбирает эти оптимумы.

## Сборка (Linux + CUDA)

Проверено на Ubuntu 26.04, RTX 3060 Laptop 6 GB (системный CUDA toolkit 12.4 из пакетов
`nvidia-cuda-toolkit`, драйвер 595 / CUDA 13.2), gcc 15 (для nvcc pin на gcc-13).

Требования: Rust (`rustup`, stable), пакет `nvidia-cuda-toolkit` (nvcc, cudart, cublas 12 —
для сборки whisper.cpp CUDA), `libclang-common-21-dev` (встроенные заголовки clang для bindgen),
gcc-13 (host-компилятор nvcc 12.x), python3.

CUDA 13 runtime для CUDA EP ONNX Runtime (prebuilt ort-бинарник слинкован с CUDA 13,
а не с системной 12.4) лежит в `deps/cuda13/` (не требует установки в систему):
`cudart`/`cublas` 13.3 — .deb из репозитория NVIDIA (ubuntu2604), распакованные в
`deps/cuda13/sys/`, cuDNN 9.27-cu13 — wheel c PyPI. FFmpeg 9 — BtbN linux64 gpl shared,
распакован в `deps/ffmpeg-n9.0-latest-linux64-gpl-shared-9.0/` (те же сонеймы, что и
в Windows-сборке). `scripts/env.sh` собирает `LD_LIBRARY_PATH` из всего этого автоматически.

```bash
scripts/build.sh            # = cargo fetch + patch whisper DTW + cargo build --release
scripts/run.sh video.mp4    # запуск (source env.sh внутри)
```

Модели кладутся так же, в `models/` (см. «Модели» ниже). Директория `models/wav2vec2-align-ru`
не используется кодом (выравнивание `--w2v` идёт через GigaAM CTC) и копируется опционально.

## Сборка (Windows, MSVC)

Требования: Rust (stable, `x86_64-pc-windows-msvc`), Visual Studio 2022 Build Tools (C++ + Win SDK),
CUDA Toolkit 13.x (nvcc, cudart, cublas), CMake, Python 3 (для cuDNN через pip), LLVM (libclang для bindgen).

```bash
# зависимости (choco):
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
choco install cuda --version=13.3.1        # если упадёт: <installer>.exe -s nvcc_13.3 cudart_13.3 cublas_13.3
choco install llvm
pip install nvidia-cudnn-cu13

# ffmpeg 9 dev-библиотеки: скачать BtbN win64 gpl shared и распаковать в deps/
# модели: см. «Модели» ниже

scripts/build.sh            # = cargo build --release c vcvars64 + DLL-деплой
scripts/add-to-path.ps1     # добавить lighttrans в PATH пользователя (для вызова из любого места)
```

Запуск без параметров или с `-h` показывает компактную справку; если она не помещается
на экран консоли, листается по нажатию клавиши (q — выход).

**Патч whisper.cpp (обязателен для длинных файлов с DTW):** при пословных таймстампах
whisper.cpp может упасть на `WHISPER_ASSERT(filter_width < a->ne[2])` (upstream
[issue #2301](https://github.com/ggml-org/whisper.cpp/issues/2301)), если декодирование
окна завершилось слишком рано. В `scripts/` лежит `patch_whisper_dtw.sh` — он ставит
guard в вендоренный whisper.cpp (крейт `whisper-rs-sys`) и принудительно пересобирает его.
`build.sh` вызывает его автоматически.

## Docker

Multi-stage `Dockerfile` (PORTING.md Фаза 7): сборка в `nvidia/cuda:13.0.2-cudnn-devel`
(rust, nvcc для whisper.cpp, ffmpeg 9 dev, патч DTW), рантайм — только бинарник +
рантайм-библиотеки ORT/ffmpeg поверх `nvidia/cuda:13.0.2-cudnn-runtime`. Модели в
образ не зашиваются — монтируются томом. Требуется `nvidia-container-toolkit` на хосте.

```bash
docker build -t lighttrans .
# GPU (verified-политика CUDA — по умолчанию; хеши отпечатков обязательны,
# см. Фазу 4 в PORTING.md; быстрый старт — allow-unverified печатает их):
docker run --rm --gpus all -p 9898:9898 -v "$PWD/models:/app/models" \
    -e ASR_CUDA_ASSIGNMENT_POLICY=allow-unverified lighttrans
docker run --rm -p 9898:9898 -v "$PWD/models:/app/models" lighttrans --ep cpu   # без GPU
```

Лимиты `ASR_*` задаются через `--env`/`-e` (значения по умолчанию в образе — как в
источнике); env верификации CUDA-назначений передаются так же. Отпечатки этой
сборки ORT/моделей совпадают с референсными (проверено в контейнере:
CTC `b99c5fae…`, RNNT `a9c3729a…`). SIGTERM/SIGINT → draining → чистый выход 0.

## Модели

Кладутся в `models/` (можно переопределить `--models-dir`):

```
models/
├── ggml-large-v3-turbo.bin            # 1.6 GB — huggingface.co/ggerganov/whisper.cpp
├── silero_vad.onnx                    # 2.3 MB  — github.com/snakers4/silero-vad (src/silero_vad/data)
├── gigaam-v3/
│   ├── v3_e2e_ctc.onnx                # 886 MB  — huggingface.co/istupakov/gigaam-v3-onnx
│   └── v3_e2e_ctc_vocab.txt
├── gigaam-v3-rnnt/                    # опционально, для -m gigaam-v3-rnnt
│   ├── v3_e2e_rnnt_encoder.onnx       # 885 MB  — huggingface.co/istupakov/gigaam-v3-onnx
│   ├── v3_e2e_rnnt_decoder.onnx       # 4.6 MB
│   ├── v3_e2e_rnnt_joint.onnx         # 2.7 MB
│   └── v3_e2e_rnnt_vocab.txt
├── parakeet-v3/                        # опционально, для -m parakeet-v3
│   ├── encoder-model.onnx              # 42 MB + .data 2.4 GB — huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx
│   ├── decoder_joint-model.onnx        # 73 MB
│   ├── nemo128.onnx                    # 140 KB  — log-mel 128 как ONNX-граф
│   ├── vocab.txt                       # 94 KB
│   └── config.json
└── nemotron3-diarization/             # опционально, для --diarize
    ├── diar_step.onnx                 # 380 MB  — ядро nvidia/Nemotron-3-Diarization
    ├── mel_filters.f32                # 129 KB  — librosa slaney filterbank (экспорт)
    ├── silence_embeds.f32             # 2 KB    — параметр модели (экспорт)
    └── meta.json                      # конфиг чанкинга/streaming

Файлы `diar_step.onnx` / `mel_filters.f32` / `silence_embeds.f32` / `meta.json`
создаются один раз скриптом `scripts/export_nemotron_diar.py`
(нужен python с torch + transformers + librosa; он же проверяет числа против PyTorch).

## Использование

```bash
scripts/run.sh --help
scripts/run.sh запись.mp4 -o out                          # gigaam-v3 по умолчанию
scripts/run.sh запись.mp4 -m gigaam-v3 -o out
scripts/run.sh запись.mp4 -m gigaam-v3-rnnt --diarize -o out   # RNN-T + спикеры
scripts/run.sh interview.mp4 -m whisper-turbo -l auto --beam-size 5
scripts/run.sh lecture.mp4 -m whisper-turbo -l ru --w2v    # быстрые пословные таймстампы (GigaAM-выравнивание)
scripts/run.sh lecture.mp4 -m whisper-turbo -l ru --dtw    # DTW-таймстампы (медленнее, точнее)
scripts/run.sh lecture.mkv -m whisper-turbo --task translate -f srt,json
scripts/run.sh interview.wav -m parakeet-v3 --diarize   # 25 языков + спикеры
scripts/run.sh noisy_interview.mp4 --denoise             # DeepFilterNet 3: подавление шума перед распознаванием
scripts/run.sh a.mp4 b.mp4 c.mp3 -m gigaam-v3 --workers 3     # несколько файлов за запуск
scripts/run.sh panel.wav --split-channels --turns -o out # каждый канал — спикер, реплики диалога
./target/release/lighttrans vad запись.mp4 -o speech.json     # интервалы речи Silero (json)
./target/release/lighttrans bench запись.mp4 -m gigaam-v3     # этапы и RTF (json в stdout)

# HTTP-сервис (после scripts/build.sh); лимиты и верификация — см. PORTING.md
ASR_CUDA_CTC_ASSIGNMENT_SHA256=<64 hex> ASR_CUDA_RNNT_ASSIGNMENT_SHA256=<64 hex> \
  ./target/release/lighttrans-server --port 9898
curl -s --data-binary @audio.mp3 "http://127.0.0.1:9898/v1/transcribe?model=gigaam-v3&words=true"
```

Основные флаги: `-m` модель (`gigaam-v3` — по умолчанию/`gigaam-v3-rnnt`/`whisper-turbo`/`parakeet-v3`), `-d` устройство (`cuda`/`cpu`),
`-l` язык (`auto`), `--task transcribe|translate`, `--diarize` (разделение спикеров),
`--w2v`/`--dtw` — выравнивание слов для whisper (CTC GigaAM / DTW; без ключей — без выравнивания),
`--denoise` — шумоподавление DeepFilterNet 3 (CPU/tract, модель вшита в бинарник;
декодирование идёт в 48 кГц, шумодав опускает сигнал в 16 кГц; `--denoise-atten-lim dB` —
предел ослабления шума, по умолчанию 100 = без ограничения). Поток делится на 2-минутные
полосы, обрабатываемые параллельно пулом воркеров прямо во время декодирования
(промежуточные файлы не создаются, в памяти держится лишь несколько полос;
~RTF 0.02 на 12-ядерном CPU, 93 мин видео ≈ 2 мин); `--denoise-threads N` — число
потоков (0 = авто, по числу ядер),
`--no-vad`, `--chunk-size 30`,
`--vad-onset 0.5 --vad-offset 0.363`, `--beam-size 1`, `--workers 0(=авто)`,
`-f` форматы вывода, `-o` каталог, `-q` тихо.
Логи: `RUST_LOG=info` (по умолчанию), `RUST_LOG=debug` — подробнее.

`--w2v` / `--dtw` управляют пословными таймстампами whisper (см. выше).
Бенчмарк на видео 93 мин: 306 с с DTW → **147 с с --w2v** — быстрее WhisperX
(214 с на том же файле). Двухфазный конвейер: сперва вся транскрибация без
DTW (с flash attention), воркеры выгружаются, затем выравнивание GigaAM-CTC
идёт на освободившейся GPU в 2 потока (~15 с на 93 мин аудио). По качеству
`--w2v` тоже точнее DTW: тайминги слов практически совпадают с нативными
таймстампами GigaAM (медиана разницы старта 0.00 с), тогда как DTW в среднем
сдвигает концы слов на ~0.24 с (растягивает в паузы).

## Архитектура

```
src/
├── audio.rs      — libav* декод: любой файл → поток блоков mono f32 (многопоточный декодер; 16 kHz или 48 kHz при --denoise)
├── df.rs         — DeepFilterNet 3 шумоподавление (крейт deep_filter/tract, модель вшита) — опция --denoise; потоковое параллельное подавление полосами прямо во время декодирования
├── mel.rs        — log-mel 64×T (n_fft 320 / hop 160 / HTK, бит-в-бит как torchaudio)
├── vad.rs        — Silero v5 ONNX (CPU), стриминг + батчи по 64 окна, гистерезис WhisperX
├── segments.rs   — стриминг-чанкер (≤ chunk_size) + merge_chunks
├── gigaam.rs     — ONNX Runtime CUDA EP, CTC greedy, пословные таймстампы из фреймов
├── gigaam_rnnt.rs — GigaAM v3 RNN-T: encoder + LSTM-prediction + joint, greedy RNN-T
├── stats.rs      — реестр этапов (старт/длительность) + сводная таблица в конце
├── parakeet.rs   — Parakeet TDT v3: mel-ONNX, FastConformer-энкодер, greedy TDT-декод
├── whisper.rs    — whisper.cpp CUDA, DTW token timestamps → слова, границы сегментов из слов
├── align.rs      — форс-выравнивание слов (CTC Viterbi по эмиссиям GigaAM) — опция --w2v, отдельная фаза после транскрибации
├── diarize.rs    — Nemotron-3-Diarization ONNX: mel-128, AOSC-кэш, offline-чанки, постпроцессинг
├── output.rs     — json (WhisperX-схема) / srt / vtt / tsv / txt
├── models.rs     — пути к моделям
```

Конвейер: один поток декодирует файл и нарезает чанки; N воркеров берут чанки из очереди
и считают на GPU; результаты собираются строго по порядку чанков. При `--w2v` после
транскрибации идёт вторая фаза — форс-выравнивание на освободившейся GPU (2 сессии GigaAM-CTC).

## Roadmap

- [x] Диаризация спикеров (Nemotron-3-Diarization → ONNX + привязка к словам)
- [x] Третья модель: Parakeet TDT 0.6B v3 (25 языков, TDT-таймстампы)
- [x] Четвёртая модель: GigaAM v3 E2E RNN-T (трансьюзер-голова)
- [x] Сводная статистика этапов по времени
- [ ] Переназначение меток спикеров по кластерам (при >2 спикерах модель может дробить)
- [x] Форс-выравнивание wav2vec2 (как в WhisperX) для whisper без DTW (`--w2v`)
- [ ] Batched-инференс GigaAM (несколько чанков за прогон ONNX)
- [ ] Автозагрузка моделей с HuggingFace (hf-hub)
