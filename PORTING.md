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
4. Сервис делать **отдельным бинарником** `lighttrans-serve` (`[[bin]]` в
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
- `dialog`/многоканальность — **не** переносить сейчас: `audio.rs` жёстко mono,
  это отдельный проект (см. «Не переносить»).

**Критерий.** `lighttrans vad file.mp4 -o out` пишет интервалы; `lighttrans
bench` печатает таблицу этапов и RTF.

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
- Многоканальный диалог/turns (`crates/transcription/src/{dialog,turns,
  channel_selection}.rs`) — LightTrans mono (`audio.rs`); отдельный проект с
  переделкой декодера.
- Монолит из 7 крейтов и типизированные числовые конверсии (`primitives`) —
  LightTrans остаётся одним крейтом; вернуться к вопросу, когда `src/` разрастётся.

## Статус

- [ ] Фаза 1 — `POST /v1/transcribe` + graceful shutdown + лимиты
- [ ] Фаза 2 — `/health` `/readyz` `/metrics`
- [ ] Фаза 3 — манифесты моделей (sha256 + IO-проверка)
- [ ] Фаза 4 — выбор EP + верификация CUDA-назначений
- [ ] Фаза 5 — CLI `bench`, `vad`
- [ ] Фаза 6 — WebSocket-стриминг (MVP mono/ctc/words)
- [ ] Фаза 7 — Docker
