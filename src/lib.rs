//! lighttrans — быстрая GPU-транскрибация аудио/видео (схема WhisperX).
//!
//! Библиотечный корень: модули общие для CLI-бинарника (`lighttrans`) и
//! сервера (`lighttrans-server`). Стриминговый конвейер: декод ffmpeg →
//! (опц. шумоподавление) → VAD → чанкер → пул GPU-воркеров → слияние.

pub mod align;
pub mod audio;
pub mod channels;
pub mod df;
pub mod diarize;
pub mod gigaam;
pub mod gigaam_rnnt;
pub mod mel;
pub mod models;
pub mod output;
pub mod parakeet;
pub mod pipeline;
pub mod segments;
pub mod stats;
pub mod transcript;
pub mod vad;
pub mod whisper;
pub mod worker;
