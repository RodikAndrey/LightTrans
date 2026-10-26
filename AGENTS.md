# AGENTS.md — путеводитель по LightTrans

Документ для агентов (и людей), дорабатывающих код. README.md отвечает на вопрос
«что умеет и как быстро», этот файл — «как внутри устроено, где что менять и как
не сломать». Документацию возможностей не дублируем: базовые сведения о моделях,
флагах и производительности — в [README.md](README.md).

Планы по переносу функциональности из gigaam-v3-runtime (HTTP-сервис, метрики,
валидация моделей и т. д.) — [PORTING.md](PORTING.md); история изменений —
[CHANGELOG.md](CHANGELOG.md).

## Что это

`lighttrans` — Rust-проект (crate `lighttrans`) с двумя бинарниками: CLI
`lighttrans` (`src/main.rs`) и HTTP-сервис `lighttrans-server`
(`src/bin/`). Один стриминговый конвейер (общий, в библиотечных модулях),
четыре сменных бэкенда распознавания (GigaAM v3 CTC/RNN-T, Whisper Turbo,
Parakeet TDT v3), вокруг них — VAD, чанкер, диаризация, шумоподавление,
форс-выравнивание и запись результатов. CLI — «файл(ы) на вход — файлы на
выход», плюс подкоманды `vad` (интервалы речи) и `bench` (этапы и RTF);
сервер — batch-`POST /v1/transcribe` (GigaAM v3), метрики Prometheus,
graceful shutdown. HTTP-слой портирован из gigaam-v3-runtime — контракт и
проверенные критерии в [PORTING.md](PORTING.md).

## Ключевые команды

```bash
source scripts/env.sh              # ОБЯЗАТЕЛЬНО и для сборки, и для запуска
scripts/build.sh [cargo-аргументы] # cargo fetch + патч whisper DTW + cargo build --release
cargo check                        # быстрая проверка компиляции (после source env.sh)
scripts/run.sh video.mp4 -m gigaam-v3 -o out      # CLI-запуск
./target/release/lighttrans-server --port 9898   # HTTP-сервис (после build)
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
├── main.rs         — CLI (clap): транскрибация + подкоманды `vad`/`bench`; сборка конвейера; фазы --w2v и --diarize; вывод
├── lib.rs          — библиотечный корень: общий код CLI и сервера
├── pipeline.rs     — общий конвейер CLI и сервера: WorkerPool (очередь + ответы по чанкам), MultiStream (mix/split-каналы), отмена
├── worker.rs       — ModelKind + сборка воркеров (try_build_worker), авто-подбор по VRAM (NVML)
├── channels.rs     — «канал = спикер»: корреляция каналов, dual-mono (>0.98), turns по паузам
├── server/         — HTTP-слой lighttrans-server (axum): /v1/transcribe, admission/draining, строгие env-лимиты, query-грамматика, health/readyz/metrics
├── metrics.rs      — счётчики/гистограммы Prometheus (lighttrans_*)
├── manifest.rs     — манифесты моделей: sha256+размер файлов, IO-имена ONNX (strict, если есть)
├── ort_assignment.rs — инспекция назначения узлов ORT (API-24): CPU/CUDA-отпечатки, политика verified/allow-unverified
├── audio.rs        — libav* декод: файл/байты/reader → поток блоков mono f32 (16 кГц; 48 кГц при --denoise; все каналы при split)
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
├── transcript.rs   — общие типы результата всех бэкендов (схема WhisperX; channel/turns для многоканальности)
├── output.rs       — запись json/srt/vtt/tsv/txt
├── models.rs       — резолв путей к моделям (--models-dir → ./models → рядом с exe) + верификация манифестов
└── stats.rs        — реестр этапов + сводная таблица/машиночитаемый json в конце запуска

src/bin/lighttrans-server.rs — HTTP-сервис: загрузка пулов GigaAM, --ep без fallback, draining по SIGTERM

scripts/            — build.sh, run.sh, env.sh, patch_whisper_dtw.sh, gen_model_manifest.py, export_nemotron_diar.py, export_w2v_align.py
tests/              — генераторы эталонов (*.py), эталонные ref_*.py (torch/transformers), фикстуры, бенч-логи
models/             — веса (в git НЕ хранятся, см. README «Модели»); манифесты manifest.json коммитятся
deps/               — cuda13 runtime, ffmpeg 9 shared, референсы (в git НЕ хранится)
```

## Конвейер и инварианты

Потоковая схема (как в WhisperX, но на потоках, не на процессах):

1. **Продюсер** (`main.rs` в CLI, `server/http.rs` в сервере; один поток):
   `audio::decode_streaming` → (опц. `df::StripPool`,
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
7. **Teardown ORT/CUDA роняет процесс.** Разрушение нескольких тёплых ONNX-сессий
   при выходе периодически даёт segfault (гонка на общем CUDA-контексте).
   `lighttrans-server` поэтому завершается через `std::process::exit()` **без
   дропа** сессий — это сознательно, не убирайте exit и не добавляйте graceful
   Drop путей для тёплого стейта.
8. **Fallback провайдера запрещён в сервере.** `lighttrans-server --ep cuda`
   собирает CUDA EP с `error_on_failure` и без ветки «переключиться на CPU»:
   недоступный CUDA — отказ старта (Фаза 4 PORTING.md). CLI сохраняет старое
   поведение (CUDA → CPU fallback) — это различие осознанное.
9. **Манифесты моделей строгие.** Если рядом с семьёй моделей лежит
   `manifest.json`, любое расхождение (файл, sha256, IO-имена ONNX) отказывает
   загрузку; для RNN-T `io` — объединённая поверхность encoder+decoder+joint,
   сверяется одним вызовом. Пересобирайте манифесты
   `scripts/gen_model_manifest.py` при обновлении весов или ONNX-экспорта.

## Git

Большие бинарники вне репозитория: `models/**/*.onnx|*.bin|*.safetensors`,
`deps/`, `target/`, медиа-фикстуры `tests/`. Коммитятся: код, скрипты, документация,
мелкие конфиги моделей (yaml/json/vocab) и эталонные json/log в `tests/`.
Стиль коммитов — императив на английском (`Add …`, `Fix …`, `Stop …`).
