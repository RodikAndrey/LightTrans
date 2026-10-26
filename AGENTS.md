# AGENTS.md — путеводитель по LightTrans

Документ для агентов (и людей), дорабатывающих код. README.md отвечает на вопрос
«что умеет и как быстро», этот файл — «как внутри устроено, где что менять и как
не сломать». Документацию возможностей не дублируем: базовые сведения о моделях,
флагах и производительности — в [README.md](README.md).

Планы по переносу функциональности из gigaam-v3-runtime (HTTP-сервис, метрики,
валидация моделей и т. д.) — [PORTING.md](PORTING.md); история изменений —
[CHANGELOG.md](CHANGELOG.md).

## Что это

`lighttrans` — одиночный Rust-бинарник (crate `lighttrans`, `src/main.rs`):
GPU-транскрибация аудио/видео по схеме WhisperX. Один стриминговый конвейер,
четыре сменных бэкенда распознавания (GigaAM v3 CTC/RNN-T, Whisper Turbo,
Parakeet TDT v3), вокруг них — VAD, чанкер, диаризация, шумоподавление,
форс-выравнивание и запись результатов. Сервисного слоя (HTTP/очереди) нет:
CLI-инструмент «файл(ы) на вход — файлы на выход».

## Ключевые команды

```bash
source scripts/env.sh              # ОБЯЗАТЕЛЬНО и для сборки, и для запуска
scripts/build.sh [cargo-аргументы] # cargo fetch + патч whisper DTW + cargo build --release
cargo check                        # быстрая проверка компиляции (после source env.sh)
scripts/run.sh video.mp4 -m gigaam-v3 -o out      # запуск
RUST_LOG=debug scripts/run.sh ...                 # подробные логи (по умолчанию info)
```

- `scripts/env.sh` нельзя «просто запустить» — его нужно **source'ить**: он задаёт
  `FFMPEG_DIR`, `CUDA_PATH`, `CUDAHOSTCXX`/`NVCC_CCBIN` (pin на gcc-13), `LIBCLANG_PATH`
  и собирает `LD_LIBRARY_PATH` (CUDA 13 runtime для CUDA EP ORT из `deps/cuda13`,
  cuDNN 9, ffmpeg, каталог ort-sys).
- `scripts/build.sh` на Linux вызывает `scripts/patch_whisper_dtw.sh` **всегда**;
  при ручном `cargo build` на свежей машине патч whisper.cpp не применён — см. «Ловушки».

## Структура

```
src/
├── main.rs         — CLI (clap, struct Args), сборка конвейера: продюсер-поток → очередь → N GPU-воркеров → слияние; фазы --w2v и --diarize; вывод
├── audio.rs        — libav* декод: любой файл → поток блоков mono f32 (16 кГц; 48 кГц при --denoise)
├── mel.rs          — log-mel 64×T, бит-в-бит как torchaudio MelSpectrogram (n_fft 320 / hop 160 / HTK)
├── vad.rs          — Silero v5 ONNX (CPU), стриминг, батчи по 64 окна, гистерезис WhisperX (onset/offset 0.5/0.363)
├── segments.rs     — стриминг-чанкер (≤ chunk_size, семантика merge_chunks WhisperX); здесь же фикс хвоста декодера
├── gigaam.rs       — GigaAM v3 E2E CTC: ONNX Runtime CUDA EP, greedy, слова из CTC-фреймов
├── gigaam_rnnt.rs  — GigaAM v3 E2E RNN-T: encoder + LSTM-prediction + joint
├── parakeet.rs     — Parakeet TDT 0.6B v3: mel-ONNX + FastConformer + greedy TDT
├── whisper.rs      — whisper.cpp CUDA (ggml-large-v3-turbo), DTW token timestamps
├── align.rs        — форс-выравнивание слов CTC-моделью GigaAM (Viterbi по эмиссиям) — фаза --w2v
├── diarize.rs      — Nemotron-3-Diarization ONNX: mel-128, AOSC-кэш (порт из HF transformers)
├── df.rs           — DeepFilterNet 3 шумоподавление (крейт deep_filter/tract, модель вшита)
├── transcript.rs   — общие типы результата всех бэкендов (схема WhisperX)
├── output.rs       — запись json/srt/vtt/tsv/txt
├── models.rs       — резолв путей к моделям (--models-dir → ./models → рядом с exe)
└── stats.rs        — реестр этапов + сводная таблица в конце запуска

scripts/            — build.sh, run.sh, env.sh, patch_whisper_dtw.sh, export_nemotron_diar.py, export_w2v_align.py
tests/              — генераторы эталонов (*.py), эталонные ref_*.py (torch/transformers), фикстуры, бенч-логи
models/             — веса (в git НЕ хранятся, см. README «Модели»)
deps/               — cuda13 runtime, ffmpeg 9 shared, референсы (в git НЕ хранится)
```

## Конвейер и инварианты

Потоковая схема (как в WhisperX, но на потоках, не на процессах):

1. **Продюсер** (`main.rs`, один поток): `audio::decode_streaming` → (опц. `df::StripPool`,
   полосы 2 минуты параллельно) → `vad::SileroVad::feed` → `segments::Chunker::feed`
   → готовые чанки в `chunk_tx`.
2. **N GPU-воркеров** берут чанки из очереди; число задаёт `--workers` (0 = авто по
   свободной VRAM через NVML).
3. **Слияние**: результаты собираются строго по порядку чанков — порядок слов и
   тайминги не должны зависеть от числа воркеров. Любой новый этап не имеет права
   нарушать этот инвариант.
4. Опциональные фазы: `--diarize` (параллельно транскрибации), `--w2v` (вторая фаза
   после транскрибации: воркеры выгружаются, затем 2 сессии GigaAM-CTC выравнивают слова).

Семантика `stats.rs`: строки сводной таблицы («инференс …, N чанков») — это
**окна-накопители** (сумма времён по чанкам, окно перекрывается с соседними этапами),
а не эксклюзивные спаны. Для точного профилирования GPU не используйте их напрямую.

## Соглашения кода

- Комментарии, тексты ошибок (`anyhow::context`) и логи `tracing` — на русском;
  идентификаторы и публичные API — на английском.
- Новый этап пайплайна обязан регистрироваться в `stats.rs` (`span`/счётчики) —
  сводная таблица в конце запуска должна покрывать весь конвейер.
- Запиненные зависимости, не обновлять походя: `ort = "=2.0.0-rc.13"`,
  `tract-* = "=0.21.4"` (совместимость libDF/deep_filter), `Cargo.lock` в git.
- Бинарные таблицы признаков (`.f32`) читаются модулями-потребителями — формат
  описан в докомментарии соответствующего модуля (`mel.rs`, `diarize.rs`), не
  изобретайте собственный сериализатор.
- Мелоксичность: числа фичи (mel, вероятности диаризации) валидируются против
  эталонов `tests/ref_*.py`; меняя вычисления признаков — прогоните сверку.

## Тесты и валидация

Единого раннера нет; `tests/` — скрипты-эталоны (нужен python с numpy/torch по ситуации):

| Скрипт | Что проверяет |
|---|---|
| `gen_dialog.py`, `gen_word_timing_test.py` | генерируют синтетические фикстуры (диалоги, TTS-слова на известных позициях) |
| `check_word_timing.py` | точность пословных таймстампов против синтетического эталона |
| `cross_check.py` | кросс-сверка таймингов двух независимых движков (CTC GigaAM vs DTW whisper.cpp) |
| `check_diarization.py` | диаризация против эталона HF transformers (мел, вероятности, привязка к словам) |
| `ref_*.py` | эталонные питон-реализации (gigaam-rnnt, nemotron, parakeet) для сверки чисел |

Медиа-фикстуры (`*.mp3/*.wav/*.mp4/*.npy`) в git не хранятся — регенерируются
скриптами; эталонные json/log-файлы коммитятся.

## Ловушки (грабли, на которые уже наступали)

1. **Хвост декодера задваивается.** После флаша декодера ffmpeg количество
   «поданных» сэмплов (`Chunker.fed`) может обгонять фактический буфер (`buf`).
   Любой срез по `fed` в чанкере обязан ограничиваться длиной буфера:
   `min(fed, buf.len())`. Проявлялось как `range end index … out of range for slice`
   в `Chunker::finish` на opus-файле ~112 минут (исправлено, см. CHANGELOG;
   clamp в `process_intervals` стоял, в `finish` — нет).
2. **whisper.cpp DTW assert** (`WHISPER_ASSERT(filter_width < a->ne[2])`, upstream
   issue #2301) — валит процесс на длинных файлах с пословными таймстампами.
   Лечится патчем `scripts/patch_whisper_dtw.sh`; `scripts/build.sh` ставит его
   автоматически, ручная сборка — нет.
3. **Два CUDA в одной системе.** CUDA EP ONNX Runtime слинкован с CUDA 13
   (`deps/cuda13/`), whisper.cpp собирается системным nvcc 12.x. `env.sh` кладёт
   CUDA 13 runtime в `LD_LIBRARY_PATH` первым и осознанно: не «упрощайте» порядок
   и не заменяйте на системный cudart.
4. **nvcc 12.x не принимает gcc ≥ 14** как host-компилятор — `env.sh` пинит
   `CUDAHOSTCXX`/`NVCC_CCBIN` на gcc-13; не убирайте этот pin.
5. **VRAM и воркеры.** Автовыбор `--workers` смотрит свободную VRAM (NVML) и на
   6 ГБ не ставит третий воркер не случайно: он проваливается в shared memory и
   замедляет чанк в ~5 раз. В бенчмарках не задавайте воркеров наугад.
6. **Пороги VAD** (0.5/0.363, min_duration 0.1 c) повторяют WhisperX/pyannote —
   это осознанная совместимость; тихие начала фраз VAD может съедать (known
   trade-off), менять — только с прогоном `check_word_timing.py`.

## Git

Большие бинарники вне репозитория: `models/**/*.onnx|*.bin|*.safetensors`,
`deps/`, `target/`, медиа-фикстуры `tests/`. Коммитятся: код, скрипты, документация,
мелкие конфиги моделей (yaml/json/vocab) и эталонные json/log в `tests/`.
Стиль коммитов — императив на английском (`Add …`, `Fix …`, `Stop …`).
