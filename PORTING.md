# PORTING.md — перенос функциональности gigaam-v3-runtime в LightTrans

Документ для отдельного агента/разработчика: что уже реализовано в
[/root/gigaam-v3-runtime](/root/gigaam-v3-runtime), чего нет в LightTrans, и как
это переносить. Контракты источника описаны в его документации и коде — **читать
их, а не придумывать**; в этом файле — карта и пометки для реализации.

## Источник и его документация

Проект: `/root/gigaam-v3-runtime` — модульный монолит (7 крейтов) вокруг GigaAM v3
(CTC + RNN-T, только ru), с сервисным слоем. Что читать перед работой:

| Документ источника | Что там |
|---|---|
| `README.md` — §What it provides | обзор возможностей, таблица крейтов |
| `docs/API.md` | контракты `POST /v1/transcribe`, `GET /v1/stream`, офлайн-CLI |
| `docs/OPERATIONS.md` | CLI сервиса, все env-переменные, admission/lifecycle, метрики, Docker |
| `docs/ARCHITECTURE.md` | граф зависимостей крейтов, границы владения |
| `docs/MODELS.md` | схема model-package v1 (config.kv, закрытый инвентарь ключей) |

Правила работы:

1. Фазы ниже независимы; порядок по ценности: 1 → 2 → 3 → 4 → 5 → 6 → 7.
   Одна фаза = один коммит/PR, не смешивать.
2. Переносим **контракты и смысл**, не код 1:1: у LightTrans другой рантайм
   (4 бэкенда распознавания, libav-декодер, один крейт). Поведение сверяем по
   докам источника.
3. Соглашения LightTrans — см. [AGENTS.md](AGENTS.md): русские комментарии/логи,
   `anyhow`/`tracing`, новый этап регистрируется в `stats.rs`, запиненные
   `ort`/`tract` не трогать.
4. Сервис делать **отдельным бинарником** `lighttrans-server` (`[[bin]]` в
   Cargo.toml), CLI `lighttrans` не ломать; общий код — модули `src/`.
5. Стек сервера взять как у источника (`crates/service/Cargo.toml`):
   `axum 0.8` (feature `ws`) + `tokio 1.x` (rt-multi-thread, signal, sync, time) —
   тогда паттерны и контракты переносятся без сюрпризов.
6. После каждой фазы: `cargo check`, прогон `scripts/run.sh` (CLI жив),
   запись в `CHANGELOG.md` (`[Unreleased]`).

## Фаза 1 — HTTP batch: `POST /v1/transcribe` (максимальная ценность)

**Зачем.** LightTrans принимает аудио только из argv. Проверено на практике:
против источника файл на 112 минут уходит одним `curl --data-binary` — этого
нам и не хватает.

**В источнике.**
- Контракт: `docs/API.md` §HTTP batch. Тело = сырые байты файла (Content-Type не
  выбирает декодер); query: `model` (ctc|rnnt), `ext`, `words`, `segments`,
  `turns`, `turn_gap`, `channels`; ответ: `duration_sec`, `sample_rate_in`,
  `channels`, `dual_mono`, `model`, `rtf` + единственная текстовая проекция
  (`text`), опционально `words`/`segments`. Коды: 400 (вход/домен), 413 (тело),
  503 (перегрузка/draining/очередь воркера >60 c), 504 (deadline). Заголовок
  `x-request-id` — на каждом ответе.
- Лимиты и политика: `docs/OPERATIONS.md` §Service limits and policy environment
  (`ASR_MAX_CONCURRENCY=8`, `ASR_BODY_LIMIT_MB=64`, `ASR_REQ_TIMEOUT_SEC=120`,
  отдельный queue-wait 60 c → 503). Значение по умолчанию ≠ fallback: кривое
  значение env **отказывает**, а не подменяется дефолтом — сохранить это
  свойство.
- Реализация: `crates/service/src/{router.rs,http.rs,admission.rs,config.rs,
  health.rs,lifecycle.rs}`.

**Как маппится на LightTrans.**
- Главное архитектурное изменение: сейчас `main.rs` живёт по схеме «загрузил
  модели → один прогон → выход». Сервису нужен тёплый стейт: ONNX-сессии
  (gigaam ctc/rnnt, silero) создаются один раз и живут между запросами;
  запрос = декодировать тело → VAD → чанки → существующие GPU-воркеры.
  Вынести «обработать поток сэмплов» в переиспользуемую функцию
  (`src/pipeline.rs`), из неё собрать и CLI-прогон, и сервис.
- Вход: `audio.rs` декодирует из пути. Добавить декод из буфера/reader
  (libavformat custom AVIO; временный файл — допустимый fallback v1).
- ORT-сессии не `Sync`: раздавать чанки через существующую схему очереди
  воркеров; **не** создавать сессию на запрос.
- Graceful shutdown по `docs/OPERATIONS.md` §Admission: SIGTERM/SIGINT →
  Draining (новые запросы 503, `/readyz` 503, активные дожидаются ≤10 c,
  обратного перехода нет). `tokio::signal` + `with_graceful_shutdown`.
- Кооперативная отмена: 504/разрыв соединения прекращает потребление новых
  чанков; при этом не утверждать, что ORT прерывает уже идущий нативный вызов
  (таков контракт источника, §Refusal and timeout behavior).

**Критерии приёмки.**
- `curl --data-binary @audio1.opus 'http://127.0.0.1:9898/v1/transcribe?model=gigaam-v3-rnnt&words=true'`
  → JSON схемы WhisperX (`transcript.rs`).
- Тело больше лимита → 413; больше `ASR_MAX_CONCURRENCY` → 503; SIGTERM во время
  запроса дожидается его завершения.

## Фаза 2 — health/readiness + Prometheus-метрики

**В источнике.** Полный список счётчиков/гейджей/гистограмм — `docs/OPERATIONS.md`
§Metrics; рендер — `crates/service/src/metrics.rs`, ручки — `health.rs`.

**Что делать.** `/health` (liveness ok), `/readyz` (503 при draining или
остановленном воркере — различать причину в теле), `/metrics` — текстовый рендер
Prometheus **без внешних крейтов** (как в источнике): фиксированные бакеты
0.01/0.025/0.05/0.1/0.25/0.5/1/2/5/10/+Inf, приём только конечных неотрицательных
наблюдений, счётчик отказов `_rejected_total`. Минимальный набор для LightTrans:
`transcribe_requests_total`, `transcribe_errors_total`, `overload_total`,
`timeout_total`, гейдж активных запросов, гистограммы `transcribe_latency`,
`frontend/encoder/decode` — снимать из тех же мест, что и `stats.rs`.

**Критерий.** После двух запросов `/metrics` показывает увеличившиеся счётчики и
непустые гистограммы; `/readyz` во время Draining отдаёт 503.

## Фаза 3 — манифест моделей: контрольные суммы + проверка IO тензоров

**В источнике.** `crates/model-package/` (`definition.rs`, `schema.rs`,
`assets.rs`, `compatibility.rs`): типизированная схема v1 с закрытым инвентарём
ключей `config.kv`, валидация пакета и выбранных артефактов **до** их
использования; схема описана в `docs/MODELS.md`.

**Адаптация.** Полную схему v1 не переносить — у LightTrans 4 семейства моделей
из разных источников. Перенести идею:
- для каждого `models/<family>/` — манифест (yaml/json): список файлов + sha256 +
  ожидаемые IO-имена/формы ключевых тензоров;
- при загрузке: проверка хешей и сверка фактических input/output имён ONNX-графа
  с манифестом; несоответствие = отказ старта с понятным сообщением
  (файл, ожидаемое, фактическое).

**Мотивация (уже обжигались).** Рассинхрон имён тензоров ловится в рантайме ORT
непонятной ошибкой (пример: `rnnt.decoder_inputs` должен быть `x,h.1,c.1`, а не
`x,hi,ci`); битые LFS-загрузки давали 29-байтовые «onnx»-файлы.

**Критерий.** Подменённый/обрезанный файл → старт не проходит, ошибка называет
файл и расхождение.

## Фаза 4 — явный выбор провайдера + верификация CUDA-назначений

**В источнике.** `crates/recognition/src/{provider.rs,ort_assignment.rs}`;
env `ASR_CUDA_ASSIGNMENT_POLICY=allow-unverified|verified`,
`ASR_CUDA_CTC_ASSIGNMENT_SHA256`, `ASR_CUDA_RNNT_ASSIGNMENT_SHA256`; описание —
`docs/OPERATIONS.md` §Provider, frontend, and trace environment и §CUDA assignment
verification, `README.md` §Validation.

**Суть.** Провайдер выбирается явно (`--ep cpu|cuda|tensorrt`), **fallback
запрещён**. Для CUDA собирается канонический отпечаток распределения узлов ORT по
execution provider'ам (роль узла → провайдер, sha256 канонического представления);
в `allow-unverified` отпечаток печатается в stderr, в `verified` — сверяется с
заданным и при несовпадении старт запрещён. Даёт воспроизводимый GPU-профиль:
без этого ORT может тихо оставить часть графа на CPU.

**В LightTrans** применимо к ONNX-бэкендам (`gigaam.rs`, `gigaam_rnnt.rs`,
`parakeet.rs`); whisper.cpp — вне схемы (помечать N/A). Отпечаток зависит от
версии ORT и файла модели — пересобирать при обновлении (референс: ORT 1.29 +
istupakov gigaam-v3, CTC `b99c5fae…`, RNNT `a9c3729a…`).

**Критерий.** `verified` с неверным хешем отказывает старт; `allow-unverified`
печатает отпечатки; при недоступном CUDA отказ вместо fallback на CPU.

## Фаза 5 — офлайн-подкоманды CLI: `bench`, `vad`

**В источнике.** `crates/cli/src/{commands/,composition.rs}`,
`docs/API.md` §Offline CLI (`asr transcribe|bench|stream|vad|dialog`).

**Что переносить.**
- `bench`: прогон файла/синтетики с поэтапным временем и RTF. База уже есть —
  оформить `stats.rs` подкомандой со стабильным машиночитаемым выводом.
- `vad`: интервалы Silero без распознавания (json). Всё нужное уже есть в
  `vad.rs`.

**Критерий.** `lighttrans vad file.mp4 -o out` пишет интервалы; `lighttrans
bench` печатает таблицу этапов и RTF.

## Многоканальный вход: каналы как спикеры (перенос из источника)

Изначально этот раздел значился в «Не переносить»; по решению владельца
переносится. Смысл: входной файл, в котором **каждый канал — отдельный спикер**
(запись стереоконференции: канал 0 = один собеседник, канал 1 = другой). Задача —
распознать каналы раздельно, а не микшировать их в mono, и собрать диалог.

**В источнике.**
- Контракт: `docs/API.md` §HTTP batch (query `channels=split`, `turns`,
  `turn_gap`; поля ответа `channels`, `dual_mono`, `channels_text`, `channel` в
  `words`/`segments`, `turns`).
- Логика: `crates/transcription/src/{channel_selection,turns,multi_batch}.rs`,
  корреляция каналов — `crates/audio/src/channel_analysis.rs`.

**Семантика (переносим как есть).**
- dual-mono: ровно 2 канала и корреляция Пирсона > 0.98 → источники совпадают;
  в split-режиме распознаётся только канал 0, в ответе `dual_mono: true`.
- `channels=split` (или запрошены `turns`) → раздельный путь: VAD/чанкер на
  каждый канал, распознавание по каналам, слова размечаются исходным номером
  канала (`channel`); канал = спикер (`SPEAKER_0N`).
- Обычный вывод (без split/turns) — прежний mono-путь LightTrans (микширование
  swr), это сознательное отличие: ниже по потоку много кода, проверенного на
  mono-сверках.
- Turns: внутри канала разрыв ≥ `turn_gap` (по умолчанию 1.0 c) между
  `word.end` и следующим `word.start` режет реплику; конец реплики — max(end)
  её слов; реплики сортируются по `(start, channel)`.

**Адаптация под LightTrans.**
- `audio.rs`: декод с сохранением каналов (target layout = исходное число
  каналов) + декод из буфера (custom AVIO `StreamIo::from_read_seek`) для
  сервера; прежний mono-путь не менять.
- Двойной-mono в стриминге: корреляция по окну первых ~30 c каждого канала
  (в источнике — `SelectionWindowSamples`); буфер окна не превышает десятки МБ.
- CLI: флаги `--split-channels`, `--turns`, `--turn-gap S`; в split-режиме
  конфликтуют с `--diarize` (канал уже даёт спикера) и `--denoise` (v1).
- Сервер: query-параметры как в источнике (`channels=split`, `turns`,
  `turn_gap`), поля ответа как в источнике.

**Критерий.** Стерео-диалог: `channels=split` даёт `channels_text` из двух
осмысленных текстов и слова с `channel 0/1`; dual-mono-файл схлопывается в один
канал с `dual_mono: true`; `turns` упорядочены по времени начала.

## Фаза 6 — WebSocket-стриминг `GET /v1/stream` (большая; отдельное решение)

**В источнике.** Контракт — `docs/API.md` §WebSocket streaming: query
(`rate`, `fmt` pcm16/f32/alaw/ulaw, `channels` 1..8, `emit` turns/words/both,
`horizon`, `endpoint`, `lock`, `turn_gap`, `dedup`), бинарные кадры = interleaved
сэмплы (границы кадров могут резать группу; неполная хвостовая группа = отказ),
квота адаптации на **канал**, а не на сокет; серверные события с хвостовыми
ревизиями (частичное/стабильное/финальное). Реализация: `crates/service/src/
{ws.rs,stream_response.rs}`, логика — `crates/transcription/src/{stream.rs,
endpoint.rs,stitch.rs,turns.rs}`.

**Что учесть.** Это не «WS-обёртка»: нужен стриминговый путь CTC — инкрементальный
mel, повторные прогоны энкодера по окнам с хвостом, ревизии гипотез, endpointing,
дедуп для многоканала. У LightTrans инкрементального пути нет вообще.
Предпосылки: фазы 1–3. MVP-срез: mono, `model=ctc`, `emit=words`, без
`turns`/`dedup`/`lock`. Оценивать отдельно.

**Критерий MVP.** Скрипт-клиент шлёт WAV по кадрам; финальная гипотеза совпадает
с batch-результатом на том же аудио в пределах допуска.

## Фаза 7 (опционально) — Docker-образ

**В источнике.** `Dockerfile.cuda` (ORT 1.29 cuda13 + `nvidia/cuda:13.0.2-cudnn`,
OCI labels), `docs/OPERATIONS.md` §Docker source builds. Учесть: LightTrans тянет
ffmpeg shared, `deps/cuda13` и nvcc-сборку whisper.cpp — делать multi-stage:
сборка в полном образе, рантайм — только рантайм-библиотеки.

## Не переносить / сознательные отличия

- TLS, аутентификация, CORS, rate-limit — в источнике их тоже нет намеренно
  (`docs/OPERATIONS.md` §Admission: сервис за ingress). Прикрывать reverse-proxy.
- Многоканальный диалог/turns — перенесён, см. «Многоканальный вход: каналы как
  спикеры»; WebSocket-дедупликация каналов (`dedup`) остаётся вне скоупа до
  решения по фазе 6.
- Монолит из 7 крейтов и типизированные числовые конверсии (`primitives`) —
  LightTrans остаётся одним крейтом; вернуться к вопросу, когда `src/` разрастётся.

## Статус

- [x] Фаза 1 — `POST /v1/transcribe` + graceful shutdown + лимиты
- [x] Многоканальный вход: каналы как спикеры (CLI + HTTP)
- [x] Фаза 2 — `/health` `/readyz` `/metrics`
- [x] Фаза 3 — манифесты моделей (sha256 + IO-проверка)
- [x] Фаза 4 — выбор EP + верификация CUDA-назначений
  (референсные отпечатки этой машины: CTC `b99c5fae…`, RNNT `a9c3729a…`,
  ORT 1.29 + istupakov gigaam-v3 — совпали с источником)
- [x] Фаза 5 — CLI `bench`, `vad`
- [x] Фаза 6 — WebSocket-стриминг (MVP mono/ctc/words): `/v1/stream`, события
  `words`/`stable`/`final`, endpointing «blank», аренда воркера CTC-пула.
  Финальная гипотеза совпала с batch (pcm16/f32/alaw); turns/lock/`endpoint=vad`/
  многоканал — 400 до upgrade (вне MVP-среза)
- [x] Фаза 7 — Docker (multi-stage `Dockerfile`: devel-сборка →
  `nvidia/cuda:13.0.2-cudnn-runtime` + рантайм-библиотеки; модели томом;
  собран и проверен на этой машине: /health, batch-запрос, GPU-проброс)
