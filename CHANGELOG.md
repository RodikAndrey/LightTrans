# Changelog

Все значимые изменения проекта LightTrans документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
и проект придерживается [семантического версионирования](https://semver.org/lang/ru/).

## [Unreleased]

### Added

- Docker-образ (Фаза 7 PORTING.md, `Dockerfile` + `.dockerignore`): multi-stage —
  сборка в `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` (rustup stable, nvcc для
  whisper.cpp, ffmpeg 9 shared dev из BtbN, патч DTW), рантайм —
  `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04` с бинарником `lighttrans-server`
  (+ CLI), библиотеками ORT (из OUT_DIR ort-sys) и рантаймом ffmpeg; CUDA 13/cuDNN 9
  берутся из базового образа (deps/cuda13 нужен только на хосте с системным CUDA 12).
  Модели не зашиваются — монтируются томом `-v $PWD/models:/app/models`; запуск
  `docker run --gpus all -p 9898:9898 … lighttrans`, CPU — `--ep cpu`. Непривилегированный
  пользователь 10001, лимиты ASR_* в ENV, OCI-метки. Кэш-маунты BuildKit для
  cargo registry/target ускоряют пересборку.
- `GET /v1/stream` — WebSocket-стриминг (Фаза 6 PORTING.md, MVP-срез: mono,
  `model=ctc`, `emit=words`). Query как у источника: `rate` обязателен
  (8000..192000, отношение к модельной ≤ 1000), `fmt` pcm16|f32|alaw|ulaw,
  `horizon`/`endpoint`/`lock`/`turn_gap`/`dedup`/`backchannel_max_ms`
  разбираются строго; значения вне MVP-среза — 400 JSON до upgrade без
  всякой работы с моделями. Бинарные кадры = сырые интерливнутые сэмплы
  (граница кадра может резать группу, терминальный неполный хвост = отказ),
  ресемплинг входа в 16 кГц с сохранением состояния; кадр/сообщение > 16 МиБ
  — close 1009. События сервера: `words` (ревизия с `revise_from`),
  `stable`, `final` (`endpoint` — финальность по endpointing), `error`,
  `end`; единственный control — `{"type":"end"}`. Инкрементальный путь
  (src/stream.rs): log-mel по фреймам, повторные прогоны энкодера по окну
  30 c с шагом 0.5 c (хвост за реальным аудио — детерминированный шум,
  seed 0x5eed источника), сшивка окон по шву (совпадение нормализованных
  слов в перекрытии), endpointing по маске тишины CTC (правила источника
  1.2/2.4 c). Воркер CTC-пула арендуется под сессию эксклюзивно; draining
  закрывает активные сессии с флашем (`final`+`end`+close), новые — 503;
  метрики `lighttrans_ws_*`.
- Многоканальный вход «каждый канал — отдельный спикер» (перенос из gigaam-v3-runtime,
  см. PORTING.md): `--split-channels` распознаёт каналы раздельно (раздельные VAD/чанкер),
  слова и сегменты получают поля `channel` и метки `SPEAKER_0N`; dual-mono-источники
  (корреляция каналов > 0.98) схлопываются в канал 0. `--turns`/`--turn-gap S` строят
  реплики диалога: разрыв ≥ `turn-gap` внутри канала режет реплику, реплики
  упорядочены по (start, channel); в txt/json — строки `[channel N]`.
  Флаги конфликтуют с `--diarize` (канал уже задаёт спикера) и `--denoise` (v1).
- `lighttrans-server` — HTTP-сервис (новый бинарь, CLI не тронут): `POST /v1/transcribe`
  с телом = сырые байты аудио (декод из памяти через custom AVIO) и строгим query
  (`model=gigaam-v3|gigaam-v3-rnnt`, `ext`, `words`, `segments`, `turns`, `turn_gap`,
  `channels=split`); ответ `duration_sec/sample_rate_in/channels/dual_mono/model/rtf`
  + одна текстовая проекция (`text` или `channels_text`) и опциональные
  `words`/`segments`/`turns`; ошибки `{"error": ...}`: 400 (вход/домен),
  413 (тело), 503 (перегрузка/draining/очередь >60 c), 504 (дедлайн);
  `x-request-id` на каждом ответе. Тёплый стейт: ONNX-сессии создаются один раз
  (пул воркеров на модель), на запрос — не создаются. Лимиты окружением
  `ASR_MAX_CONCURRENCY`/`ASR_BODY_LIMIT_MB`/`ASR_REQ_TIMEOUT_SEC` (кривое значение
  отказывает, не подменяется); SIGTERM/SIGINT → draining (новые 503, активные дожидаются).
- `AGENTS.md` — путеводитель по проекту для агентов: структура, команды сборки/запуска,
  инварианты конвейера, соглашения кода, тесты и известные ловушки.
- `CHANGELOG.md` — этот файл.
- Подкоманды CLI (Фаза 5): `lighttrans vad <files> [-o PATH]` — интервалы речи
  Silero VAD без распознавания (json: длительность, пороги, интервалы; по умолчанию
  `<вход>.vad.json`, при нескольких входах `-o` — каталог); `lighttrans bench <files>`
  — полный прогон транскрибации с машиночитаемой сводкой в stdout (json:
  `wall_sec`/`audio_sec`/`rtf`/`files`/`stages` — те же окна-накопители, что в
  таблице; логи и таблица при этом уходят в stderr). Запуск без подкоманды
  работает как раньше.
- Манифесты моделей (Фаза 3): `manifest.json` рядом с каждой семьёй моделей
  (sha256+размер каждого файла, опционально IO-имёна ONNX-графа). Верификация при
  загрузке: битый/подменённый файл или рассинхрон IO-имён отказывает старт с
  сообщением (файл, ожидаемый и фактический хеш). Генератор —
  `scripts/gen_model_manifest.py` (`onnx`-пакет для IO).
- Явный выбор провайдера и верификация CUDA-назначений (Фаза 4): `lighttrans-server
  --ep cpu|cuda` без fallback — недоступный CUDA отказывает старт, а не переключает
  на CPU (CUDA EP собирается с `error_on_failure`). Инспекция распределения узлов
  ORT по execution provider'ам (API-24, `session.record_ep_graph_assignment_info`):
  принимаются только CPU/CUDA-назначения и требуется ≥1 CUDA-узел; канонический
  sha256-отпечаток (домен как в gigaam-v3-runtime, референсы ORT 1.29 + istupakov
  gigaam-v3: CTC `b99c5fae…`, RNNT `a9c3729a…`). Политика `ASR_CUDA_ASSIGNMENT_POLICY`:
  absent/`verified` — отпечатки обязательны и сверяются (несовпадение отказывает),
  `allow-unverified` — роль/отпечаток/счётчики узлов печатаются в stderr
  (`# CUDA assignment unverified role=… sha256=… cpu_nodes=… cuda_nodes=…`), конфликт
  с хешами — отказ. Настройки `ASR_CUDA_*` при `--ep cpu` — отказ. CLI не затронут.

## [0.1.0] - 2026-10-25

Первый зафиксированный срез (инициализация git-репозитория).

### Added

- Стриминговый GPU-конвейер транскрибации в духе WhisperX на потоках:
  декодирование через libav* (любой ffmpeg-формат, аудио и видео) → Silero VAD v5
  (гистерезис WhisperX) → нарезка на чанки ≤ `--chunk-size` → N GPU-воркеров
  (число по свободной VRAM через NVML, `--workers`) → слияние результатов строго
  по порядку чанков. Несколько входных файлов за запуск, промежуточных файлов нет.
- Четыре модели распознавания: GigaAM v3 E2E CTC (по умолчанию), GigaAM v3 E2E
  RNN-T, Whisper Large v3 Turbo (whisper.cpp CUDA: `--task translate`, выбор языка,
  beam), Parakeet TDT 0.6B v3 (25 языков, таймстампы из TDT).
- Пословные таймстампы со score: из CTC-фреймов (GigaAM, Parakeet) и DTW token
  timestamps (whisper, `--dtw`).
- Форс-выравнивание слов `--w2v` для whisper-turbo: двухфазный конвейер, CTC-модель
  GigaAM выравнивает слова на освободившейся GPU (быстрее и точнее DTW).
- Диаризация спикеров `--diarize`: NVIDIA Nemotron-3-Diarization (ядро экспортировано
  в ONNX скриптом `scripts/export_nemotron_diar.py`, AOSC-кэш портирован 1:1 из
  HF transformers); метки спикеров в словах и сегментах (json) и префиксом в srt/vtt.
- Шумоподавление `--denoise`: DeepFilterNet 3 (модель вшита в бинарник), потоковая
  параллельная обработка полосами во время декодирования, декод в 48 кГц.
- Вывод: json (схема WhisperX: segments + word_segments), srt, vtt, tsv, txt
  (`-f`, `-o`).
- Сводная статистика этапов в конце запуска (`stats.rs`): старт/длительность этапов,
  чанки, аудио-секунды, RTF.
- Сборка Linux (Ubuntu: системный nvcc 12 для whisper.cpp + prebuilt ORT CUDA EP на
  CUDA 13 runtime из `deps/cuda13`, ffmpeg 9 shared из `deps/`) и Windows MSVC
  (vcvars64, деплой DLL рядом с exe); `scripts/build.sh`, `scripts/run.sh`,
  `scripts/env.sh` для обеих платформ.
- Патч whisper.cpp `scripts/patch_whisper_dtw.sh` (guard против
  `WHISPER_ASSERT(filter_width < a->ne[2])`, upstream issue #2301) — ставится
  автоматически при сборке через `scripts/build.sh`.
- Валидационные утилиты `tests/`: точность пословных таймстампов против синтетического
  эталона (`check_word_timing.py`), кросс-сверка CTC/DTW (`cross_check.py`), сверка
  диаризации с эталоном HF transformers (`check_diarization.py`), эталонные
  питон-реализации `ref_*.py`.
- `.gitignore` для крупных бинарников (модели, `deps/`, `target/`, медиа-фикстуры).

### Fixed

- Panic `range end index … out of range for slice` в `Chunker::finish`
  (`src/segments.rs`) на длинных файлах: после флаша декодера счётчик поданных
  сэмплов `fed` может обгонять фактический буфер, из-за чего срез `buf[..close]`
  выходил за границы (воспроизводилось на opus ~112 минут). Граница закрытия
  теперь дополнительно ограничена длиной буфера — так же, как уже сделано в
  `process_intervals`.
