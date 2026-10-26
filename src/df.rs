//! Шумоподавление DeepFilterNet 3 (Rust-либа libDF из крейта `deep_filter`,
//! CPU-бэкенд tract, модель DFN3 вшита в бинарник через feature default-model).
//!
//! Пайплайн работает в 16 кГц, а модель DFN3 — в 48 кГц, поэтому при включённом
//! `--denoise` декодирование идёт сразу в 48 кГц: шумодав обрабатывает 48 кГц,
//! после чего сигнал ресемплируется вниз в 16 кГц (rubato). Задержка STFT +
//! lookahead модели и латентность ресемплёра компенсируются отбрасыванием
//! первых отсчётов выходного потока, чтобы тайминги VAD/чанкера остались
//! выровнены с исходным сигналом.
//!
//! tract (бэкенд libDF) исполняет модель однопоточно, поэтому поток делится на
//! полосы, которые обрабатываются параллельно пулом воркеров
//! ([`StripPool`]). Полоса сопровождается 6-секундным «прогревом» — хвостом
//! предыдущей полосы: константа времени нормализации модели norm_tau = 1 с,
//! за 6 с GRU-состояние и нормализация сходятся, и стыки полос получаются
//! практически неотличимы от непрерывной обработки. Каждый воркер загружает
//! модель один раз и клонирует её на каждую полосу (`DfTract: Clone`, клон
//! дешевле компиляции на порядки). Пул работает с противодавлением: пока
//! воркеры заняты, декодирование приостанавливается, поэтому в памяти
//! одновременно находится лишь несколько полос, а не весь файл.
//!
//! GPU для DFN3 неэффективен: модель крошечная и рекуррентная (3 запуска сети
//! на каждые 10 мс звука), накладные расходы на запуск ядер GPU съели бы весь
//! выигрыш.

use crate::audio::TARGET_SR;
use anyhow::{Context, Result};
use ::df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::{Array2, ArrayView2};
use rubato::{FftFixedIn, Resampler};
use std::collections::BTreeMap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Частота дискретизации модели DeepFilterNet 3.
pub const MODEL_SR: u32 = 48_000;

/// Входной чанк ресемплёра 48k -> 16k, отсчётов (100 мс).
const DOWN_CHUNK: usize = 4_800;

/// «Прогрев» полосы, отсчётов 48 кГц (6 с): константа времени нормализации
/// признаков модели norm_tau = 1 с, за 6 с состояние сходится на 99.7%.
const PRIME: usize = 6 * MODEL_SR as usize;

/// Длина полосы, отсчётов 48 кГц (120 с): компромисс между объёмом памяти
/// (полосы в полёте) и накладными расходами на прогрев (6 с на полосу).
const STRIP: usize = 120 * MODEL_SR as usize;

/// Собирает новый экземпляр модели с заданным пределом ослабления шума.
fn build_df(atten_lim_db: f32) -> Result<DfTract> {
    let rp = RuntimeParams::default().with_atten_lim(atten_lim_db);
    let df = DfTract::new(DfParams::default(), &rp)
        .context("загрузка модели DeepFilterNet 3 (вшитой)")?;
    tracing::info!(
        fft = df.fft_size,
        hop = df.hop_size,
        lookahead = df.lookahead,
        atten_lim_db,
        "DeepFilterNet 3 загружен (CPU/tract)"
    );
    Ok(df)
}

pub struct DfDenoiser {
    df: DfTract,
    down: FftFixedIn<f32>,
    down_in: Vec<Vec<f32>>,
    down_out: Vec<Vec<f32>>,
    hop: usize,
    enh: Array2<f32>,
    /// Непрошедшие обработку 48k-сэмплы (меньше полного hop).
    in48: Vec<f32>,
    /// Enhanced 48k samples awaiting a full down-chunk.
    enh48: Vec<f32>,
    delay16: usize,
    dropped: usize,
    total_in: usize,
    total_out: usize,
    elapsed: Instant,
}

impl DfDenoiser {
    /// Из готовой модели (клона нетронутого шаблона — состояние в нём чистое).
    pub fn from_df(df: DfTract) -> Result<Self> {
        assert_eq!(
            df.sr, MODEL_SR as usize,
            "модель DeepFilterNet не 48 кГц — нужна доработка df.rs"
        );
        let hop = df.hop_size;
        // задержка STFT + lookahead модели (в отсчётах 48 кГц), как в
        // примере libDF с --compensate-delay
        let delay48 = df.fft_size - df.hop_size + df.lookahead * df.hop_size;
        let ratio = MODEL_SR as usize / TARGET_SR as usize;
        let down = FftFixedIn::<f32>::new(MODEL_SR as usize, TARGET_SR as usize, DOWN_CHUNK, 1, 1)
            .context("инициализация ресемплёра 48k -> 16k")?;
        let delay16 = (delay48 as f64 / ratio as f64).round() as usize + down.output_delay();
        let down_in = down.input_buffer_allocate(true);
        let down_out = down.output_buffer_allocate(true);
        Ok(Self {
            df,
            down,
            down_in,
            down_out,
            hop,
            enh: Array2::zeros((1, hop)),
            in48: Vec::new(),
            enh48: Vec::new(),
            delay16,
            dropped: 0,
            total_in: 0,
            total_out: 0,
            elapsed: Instant::now(),
        })
    }

    /// Обрабатывает блок моно-сэмплов 48 кГц, возвращает очищенные сэмплы 16 кГц.
    pub fn process_block(&mut self, block48: &[f32]) -> Result<Vec<f32>> {
        self.total_in += block48.len();
        let mut out = Vec::with_capacity(block48.len() / (MODEL_SR as usize / TARGET_SR as usize) + 16);
        // батчами по 1с, чтобы промежуточные буферы не разрастались до размера
        // всего блока (drain с головы вектора иначе даёт O(n^2) на memmove)
        for sub in block48.chunks(MODEL_SR as usize) {
            self.in48.extend_from_slice(sub);
            let n_frames = self.in48.len() / self.hop;
            for f in 0..n_frames {
                let ns = ArrayView2::from_shape((1, self.hop), &self.in48[f * self.hop..(f + 1) * self.hop])
                    .expect("срез hop_size непрерывен");
                self.df
                    .process(ns, self.enh.view_mut())
                    .context("инференс DeepFilterNet")?;
                self.enh48.extend_from_slice(self.enh.as_slice().unwrap());
            }
            self.in48.drain(..n_frames * self.hop);
            // ресемплинг: идём курсором по enh48 и один раз уплотняем хвост
            let mut pos = 0;
            while self.enh48.len() - pos >= DOWN_CHUNK {
                self.down_in[0][..DOWN_CHUNK].copy_from_slice(&self.enh48[pos..pos + DOWN_CHUNK]);
                let (_, n_out) = self
                    .down
                    .process_into_buffer(&self.down_in, &mut self.down_out, None)
                    .context("ресемплинг 48k -> 16k")?;
                let chunk: Vec<f32> = self.down_out[0][..n_out].to_vec();
                self.emit(&chunk, &mut out);
                pos += DOWN_CHUNK;
            }
            self.enh48.drain(..pos);
        }
        Ok(out)
    }

    /// Флаш внутренних буферов; возвращает остаток очищенных сэмплов 16 кГц.
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        // хвост меньше hop — дополняем нулями (на выходе это просто тишина)
        if !self.in48.is_empty() {
            self.in48.resize(self.hop, 0.0);
            let ns = ArrayView2::from_shape((1, self.hop), &self.in48).unwrap();
            self.df
                .process(ns, self.enh.view_mut())
                .context("инференс DeepFilterNet (хвост)")?;
            self.enh48.extend_from_slice(self.enh.as_slice().unwrap());
            self.in48.clear();
        }
        // вымываем состояние ресемплёра: добиваем до целого чанка + два
        // нулевых чанка про запас
        let chunks = self.enh48.len().div_ceil(DOWN_CHUNK) + 2;
        self.enh48.resize(chunks * DOWN_CHUNK, 0.0);
        while self.enh48.len() >= DOWN_CHUNK {
            self.down_in[0][..DOWN_CHUNK].copy_from_slice(&self.enh48[..DOWN_CHUNK]);
            let (_, n_out) = self
                .down
                .process_into_buffer(&self.down_in, &mut self.down_out, None)
                .context("ресемплинг 48k -> 16k (флаш)")?;
            let chunk: Vec<f32> = self.down_out[0][..n_out].to_vec();
            self.emit(&chunk, &mut out);
            self.enh48.drain(..DOWN_CHUNK);
        }
        // добираем недостающее нулями: из-за компенсации задержки хвост
        // длительностью delay16 заменён тишиной (как --compensate-delay в libDF)
        let needed = self.total_in / (MODEL_SR as usize / TARGET_SR as usize);
        while self.total_out < needed {
            out.push(0.0);
            self.total_out += 1;
        }
        tracing::debug!(
            "шумоподавление (полоса): {:.1}с аудио за {:.2}с (RTF {:.3})",
            self.total_in as f64 / MODEL_SR as f64,
            self.elapsed.elapsed().as_secs_f64(),
            self.elapsed.elapsed().as_secs_f64()
                / (self.total_in as f64 / MODEL_SR as f64).max(1e-9),
        );
        Ok(out)
    }

    /// Кладёт очередную порцию выходных сэмплов 16 кГц: отрезает начальную
    /// задержку и не выпускает больше, чем положено по длине входа.
    fn emit(&mut self, samples: &[f32], out: &mut Vec<f32>) {
        let needed = self.total_in / (MODEL_SR as usize / TARGET_SR as usize);
        let allowed = needed.saturating_sub(self.total_out);
        let take = samples.len().min(allowed);
        let skip = (self.delay16 - self.dropped).min(take);
        self.dropped += skip;
        out.extend_from_slice(&samples[skip..take]);
        self.total_out += take - skip;
    }
}

/// Обрабатывает одну полосу (прогрев + данные) шаблоном из `tpl`; при первом
/// вызове лениво загружает модель. Возвращает очищенные сэмплы 16 кГц только
/// за данные (прогрев отбрасывается): выходной поток после компенсации
/// задержки выровнен со входом (lag 0), поэтому нужный кусок просто
/// вырезается из полного выхода.
fn run_strip(
    tpl: &mut Option<DfTract>,
    atten_lim_db: f32,
    prime48: usize,
    data: &[f32],
) -> Result<Vec<f32>> {
    if tpl.is_none() {
        *tpl = Some(build_df(atten_lim_db)?);
    }
    let mut d = DfDenoiser::from_df(tpl.as_ref().unwrap().clone())?;
    let mut raw = d.process_block(data)?;
    raw.extend(d.finish()?);
    let ratio = MODEL_SR as usize / TARGET_SR as usize;
    let skip = prime48 / ratio;
    let take = (data.len() - prime48) / ratio;
    anyhow::ensure!(
        raw.len() >= skip + take,
        "шумодав выдал {} сэмплов, ожидалось >= {}",
        raw.len(),
        skip + take
    );
    Ok(raw[skip..skip + take].to_vec())
}

type StripMsg = (u64, usize, Vec<f32>); // индекс полосы, длина прогрева, прогрев+данные (48k)
type ResMsg = (u64, Result<Vec<f32>>);  // индекс полосы, очищенные сэмплы 16k

/// Пул воркеров шумоподавления, работающий в потоковом режиме вместе с
/// декодером: [`StripPool::feed`] вызывается из колбэка декодера по мере
/// поступления блоков 48 кГц, полосы dispatched воркерам, готовые — в порядке
/// индексов отдаются в `emit` (VAD/чанкер). Противодавление не даёт буферам
/// расти: при `threads + 2` полосах в полёте декодирование приостанавливается.
pub struct StripPool {
    threads: usize,
    tx: Option<Sender<StripMsg>>,
    rx: Receiver<ResMsg>,
    /// Текущая полоса (48 кГц, без прогрева).
    cur: Vec<f32>,
    /// Хвост предыдущей полосы — прогрев для следующей.
    prime_tail: Vec<f32>,
    next_dispatch: u64,
    next_emit: u64,
    pending: BTreeMap<u64, Result<Vec<f32>>>,
    in_flight: usize,
    strips: usize,
    t0: Option<Instant>,
    total_in48: usize,
}

impl StripPool {
    /// Создаёт пул; воркеры загружают модель лениво, при первой полосе.
    pub fn new(atten_lim: f32, threads_req: usize) -> Self {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let threads = if threads_req > 0 { threads_req } else { cores };
        let (tx, rx_work) = channel::<StripMsg>();
        let (tx_res, rx_res) = channel::<ResMsg>();
        let rx_work = Arc::new(Mutex::new(rx_work));
        for _ in 0..threads {
            let rx = Arc::clone(&rx_work);
            let tx_res = tx_res.clone();
            let atten_lim = atten_lim;
            std::thread::spawn(move || {
                let mut tpl: Option<DfTract> = None;
                loop {
                    let msg = rx.lock().unwrap().recv();
                    let (idx, prime48, data) = match msg {
                        Ok(m) => m,
                        Err(_) => break, // канал закрыт — воркер завершается
                    };
                    let res = run_strip(&mut tpl, atten_lim, prime48, &data);
                    let _ = tx_res.send((idx, res));
                }
            });
        }
        Self {
            threads,
            tx: Some(tx),
            rx: rx_res,
            cur: Vec::new(),
            prime_tail: Vec::new(),
            next_dispatch: 0,
            next_emit: 0,
            pending: BTreeMap::new(),
            in_flight: 0,
            strips: 0,
            t0: None,
            total_in48: 0,
        }
    }

    /// Принимает очередной блок моно-сэмплов 48 кГц от декодера. Готовые
    /// полосы в исходном порядке передаются в `emit` (сэмплы 16 кГц).
    pub fn feed(&mut self, block48: &[f32], emit: &mut dyn FnMut(&[f32]) -> Result<()>) -> Result<()> {
        if self.t0.is_none() {
            self.t0 = Some(Instant::now());
        }
        self.total_in48 += block48.len();
        let mut rest = block48;
        while !rest.is_empty() {
            let take = rest.len().min(STRIP - self.cur.len());
            self.cur.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.cur.len() == STRIP {
                self.dispatch(emit)?;
            }
        }
        self.pump(emit)
    }

    /// Завершает поток: отправляет неполную последнюю полосу, собирает все
    /// результаты и приводит пул в состояние «готов к новому файлу».
    pub fn finish(&mut self, emit: &mut dyn FnMut(&[f32]) -> Result<()>) -> Result<()> {
        self.dispatch(emit)?;
        drop(self.tx.take()); // воркеры завершатся, разобрав оставшиеся полосы
        while self.in_flight > 0 {
            let (idx, res) = self.rx.recv().context("поток шумоподавления завершился")?;
            self.pending.insert(idx, res);
            self.in_flight -= 1;
            self.emit_ready(emit)?;
        }
        if let Some(t0) = self.t0 {
            let secs = t0.elapsed().as_secs_f64();
            let audio = self.total_in48 as f64 / MODEL_SR as f64;
            tracing::info!(
                "шумоподавление: {:.1}с аудио за {:.2}с (RTF {:.3}), полос: {}, потоков: {}",
                audio,
                secs,
                secs / audio.max(1e-9),
                self.next_dispatch,
                self.threads
            );
        }
        // сброс к новому файлу (воркеры и загруженные модели остаются живы)
        self.cur.clear();
        self.prime_tail.clear();
        self.next_dispatch = 0;
        self.next_emit = 0;
        self.pending.clear();
        self.strips = 0;
        self.t0 = None;
        self.total_in48 = 0;
        Ok(())
    }

    /// Отправляет текущую полосу воркеру, при необходимости приостановив
    /// декодирование (противодавление).
    fn dispatch(&mut self, emit: &mut dyn FnMut(&[f32]) -> Result<()>) -> Result<()> {
        if self.cur.is_empty() {
            return Ok(());
        }
        while self.in_flight >= self.threads + 2 {
            let (idx, res) = self.rx.recv().context("поток шумоподавления завершился")?;
            self.pending.insert(idx, res);
            self.in_flight -= 1;
            self.emit_ready(emit)?;
        }
        let prime48 = self.prime_tail.len();
        let mut data = std::mem::take(&mut self.prime_tail);
        data.extend_from_slice(&self.cur);
        self.prime_tail = self.cur[self.cur.len().saturating_sub(PRIME)..].to_vec();
        self.cur.clear();
        let idx = self.next_dispatch;
        self.next_dispatch += 1;
        self.tx
            .as_ref()
            .expect("канал полос закрыт до finish")
            .send((idx, prime48, data))
            .context("отправка полосы воркеру")?;
        self.in_flight += 1;
        Ok(())
    }

    /// Неблокирующе собирает готовые результаты и отдаёт их подряд.
    fn pump(&mut self, emit: &mut dyn FnMut(&[f32]) -> Result<()>) -> Result<()> {
        while let Ok((idx, res)) = self.rx.try_recv() {
            self.pending.insert(idx, res);
            self.in_flight -= 1;
            self.emit_ready(emit)?;
        }
        Ok(())
    }

    /// Отдаёт в `emit` все подряд идущие готовые полосы, начиная с next_emit.
    fn emit_ready(&mut self, emit: &mut dyn FnMut(&[f32]) -> Result<()>) -> Result<()> {
        while let Some(res) = self.pending.remove(&self.next_emit) {
            let out = res.context("ошибка в потоке шумоподавления")?;
            emit(&out)?;
            self.next_emit += 1;
            self.strips += 1;
            if self.strips % 10 == 0 {
                tracing::info!("шумоподавление: готово полос: {}", self.strips);
            }
        }
        Ok(())
    }
}
