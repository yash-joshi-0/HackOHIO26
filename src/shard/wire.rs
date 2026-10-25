//! Async codec for david-codex's v1 shard-runner protocol (`shard_runner.cpp`).
//!
//! A runner prints one JSON metadata line on startup, then serves requests in order: each
//! request is five big-endian `u32` words ([`RequestHeader`]) followed by either big-endian
//! token IDs (the first stage) or little-endian `f32` hidden rows (every later stage). Each
//! reply is one JSON line ([`StepReply`]) followed by `n_floats` little-endian `f32` values.
//! A remote client first sends [`CLIENT_PREFACE`]; a local stdio runner gets no preface.
//!
//! Both sides live here: the coordinator uses the client half (`write_request` / `read_step`),
//! and relays or fake runners use the server half (`read_request_header` / `write_step`).
//! Every length read off the wire is checked against a fixed bound before anything is
//! allocated, so a corrupt or hostile peer cannot make either side allocate unboundedly.

use anyhow::{Context, bail, ensure};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt,
};

/// Bytes a remote client sends before anything else; the endpoint stays silent until then.
pub const CLIENT_PREFACE: &[u8; 5] = b"MSH1\n";

/// First word of every request header (`"MSH1"` read as a big-endian `u32`).
pub const REQUEST_MAGIC: u32 = 0x4d53_4831;

/// Largest number of tokens (or hidden rows) one request may carry.
pub const MAX_BATCH_TOKENS: u32 = 128;

/// Largest context window the runner accepts.
pub const MAX_CONTEXT: u32 = 4096;

/// Bound on a metadata or reply JSON line, newline excluded.
pub const MAX_LINE_BYTES: usize = 1 << 20;

/// Largest hidden size the runner accepts (`shard_runner.cpp` refuses anything above it).
pub const MAX_EMBD: u32 = 32_768;

/// Bound on the hidden values in one request: a full batch of the widest supported rows.
const MAX_HIDDEN_VALUES: usize = MAX_BATCH_TOKENS as usize * MAX_EMBD as usize;

/// Number of big-endian `u32` words in a request header.
const HEADER_WORDS: usize = 5;

/// Size of a request header on the wire.
const REQUEST_HEADER_LEN: usize = HEADER_WORDS * WORD_LEN;

/// Size of one encoded `u32` token or `f32` value.
const WORD_LEN: usize = 4;

/// Lowercase digits used to encode `piece_hex`, matching the runner's output.
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

/// What a request's payload carries, fixed per stage by whether it begins at layer 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    /// Token IDs, sent to the first stage (`begin == 0`).
    Tokens = 1,
    /// Hidden-state rows, sent to every later stage (`begin > 0`).
    Hidden = 2,
}

impl TryFrom<u32> for Opcode {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> anyhow::Result<Self> {
        match value {
            code if code == Self::Tokens as u32 => Ok(Self::Tokens),
            code if code == Self::Hidden as u32 => Ok(Self::Hidden),
            other => bail!("unknown request opcode {other}"),
        }
    }
}

/// A decoded request header; the payload that follows it is read separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestHeader {
    /// Payload kind.
    pub opcode: Opcode,
    /// Sequence number: starts at 0 and increments by one per request.
    pub step: u32,
    /// Position of the first token: starts at 0 and advances by `token_count`.
    pub position: u32,
    /// Tokens (or hidden rows) in the payload, always within `1..=MAX_BATCH_TOKENS`.
    pub token_count: u32,
}

/// The payload of one request, borrowed from the caller.
pub enum StageInput<'a> {
    /// Token IDs for the first stage.
    Tokens(&'a [u32]),
    /// `token_count` rows of `embd` hidden values each, laid out row after row.
    Hidden {
        /// The rows, flattened.
        rows: &'a [f32],
        /// Hidden size: values per row.
        embd: u32,
    },
}

impl StageInput<'_> {
    /// The opcode this payload is sent with.
    pub fn opcode(&self) -> Opcode {
        match self {
            Self::Tokens(_) => Opcode::Tokens,
            Self::Hidden { .. } => Opcode::Hidden,
        }
    }

    /// Number of tokens the payload covers, checked to be within `1..=MAX_BATCH_TOKENS` and,
    /// for hidden rows, to be a whole number of rows.
    pub fn token_count(&self) -> anyhow::Result<u32> {
        let count = match self {
            Self::Tokens(tokens) => tokens.len(),
            Self::Hidden { rows, embd } => {
                ensure!(*embd > 0, "hidden rows need a nonzero embd");
                let embd = *embd as usize;
                ensure!(
                    rows.len() % embd == 0,
                    "{} hidden values are not whole rows of {embd}",
                    rows.len()
                );
                rows.len() / embd
            }
        };
        let count = u32::try_from(count).unwrap_or(u32::MAX);
        check_token_count(count)?;
        Ok(count)
    }
}

/// The runner's startup line, describing the stage it serves.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageMetadata {
    /// Protocol version (1).
    pub protocol: u32,
    /// First layer of the stage (inclusive).
    pub begin: u32,
    /// End of the stage's layer range (exclusive).
    pub end: u32,
    /// Total layers in the model.
    pub layers: u32,
    /// Hidden size.
    pub embd: u32,
    /// Vocabulary size.
    pub vocab: u32,
    /// Runtime buffer bytes the runner allocated at startup.
    #[serde(default)]
    pub allocated_bytes: u64,
    /// Original model identity embedded in an exported shard, or empty.
    #[serde(default)]
    pub source_sha256: String,
    /// The runner's tokenization of its `PROMPT` argument.
    #[serde(default)]
    pub tokens: Vec<u32>,
    /// Model identity, added by a relay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
    /// SHA-256 of the shard file actually loaded, added by a relay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_sha256: Option<String>,
    /// SHA-256 of the runner binary, added by a relay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner_sha256: Option<String>,
    /// Every other field (max_rss_bytes, vm, ...), kept so a relay can forward it untouched.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl StageMetadata {
    /// Whether this stage ends at the last layer, so it samples tokens instead of returning rows.
    pub fn is_final(&self) -> bool {
        self.end == self.layers
    }
}

/// The peer closed the stream before a whole JSON line arrived: a runner that exited or a
/// connection that dropped, as opposed to a peer that sent something malformed.
#[derive(Debug, thiserror::Error)]
#[error("stream closed {place} the {what} line")]
pub struct StreamClosed {
    /// `"before"` or `"in the middle of"`.
    place: &'static str,
    /// Which line was expected.
    what: String,
}

/// The JSON line the runner sends for each request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepReply {
    /// The request's step number.
    pub step: u32,
    /// Little-endian `f32` values following this line.
    pub n_floats: usize,
    /// Greedy token from the final stage, or -1 from any other stage.
    pub token: i64,
    /// Whether `token` ends generation.
    pub eog: bool,
    /// Hex of the token's detokenized bytes (possibly part of a UTF-8 character).
    #[serde(default)]
    pub piece_hex: String,
    /// Time spent in `llama_decode`.
    #[serde(default)]
    pub compute_ms: f64,
    /// Peak runtime buffer bytes observed so far.
    #[serde(default)]
    pub observed_peak_buffer_bytes: u64,
    /// Every other field (max_rss_bytes, vm, ...), kept so a relay can forward it untouched.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl StepReply {
    /// Decodes `piece_hex` into the raw bytes of the token's text.
    pub fn piece_bytes(&self) -> anyhow::Result<Vec<u8>> {
        decode_hex(&self.piece_hex)
            .with_context(|| format!("invalid piece_hex in step {} reply", self.step))
    }
}

/// One decoded reply: the JSON line and the values that followed it.
pub struct StepOutput {
    /// The reply line.
    pub reply: StepReply,
    /// Exactly `reply.n_floats` values.
    pub floats: Vec<f32>,
}

/// Client: send [`CLIENT_PREFACE`] to a remote endpoint.
pub async fn write_preface<W: AsyncWrite + Unpin>(w: &mut W) -> anyhow::Result<()> {
    w.write_all(CLIENT_PREFACE)
        .await
        .context("failed to send shard client preface")?;
    w.flush()
        .await
        .context("failed to flush shard client preface")
}

/// Client: read the runner's startup metadata line.
pub async fn read_metadata<R: AsyncBufRead + Unpin>(r: &mut R) -> anyhow::Result<StageMetadata> {
    read_json_line(r, "shard metadata").await
}

/// Client: send one request (header and payload) and flush it.
pub async fn write_request<W: AsyncWrite + Unpin>(
    w: &mut W,
    step: u32,
    position: u32,
    input: &StageInput<'_>,
) -> anyhow::Result<()> {
    let token_count = input.token_count()?;
    let header = [
        REQUEST_MAGIC,
        input.opcode() as u32,
        step,
        position,
        token_count,
    ];
    let mut frame = Vec::with_capacity(REQUEST_HEADER_LEN + payload_len(input));
    for word in header {
        frame.extend_from_slice(&word.to_be_bytes());
    }
    match input {
        StageInput::Tokens(tokens) => {
            for token in *tokens {
                frame.extend_from_slice(&token.to_be_bytes());
            }
        }
        StageInput::Hidden { rows, .. } => frame.extend_from_slice(&encode_f32s(rows)),
    }
    w.write_all(&frame)
        .await
        .with_context(|| format!("failed to send shard request for step {step}"))?;
    w.flush()
        .await
        .with_context(|| format!("failed to flush shard request for step {step}"))
}

/// Client: read one reply, refusing one that carries more than `max_floats` values.
pub async fn read_step<R: AsyncBufRead + Unpin>(
    r: &mut R,
    max_floats: usize,
) -> anyhow::Result<StepOutput> {
    let reply: StepReply = read_json_line(r, "shard step reply").await?;
    ensure!(
        reply.n_floats <= max_floats,
        "step {} reply carries {} floats, more than the expected {max_floats}",
        reply.step,
        reply.n_floats
    );
    reply.piece_bytes()?;
    let floats = read_f32s(r, reply.n_floats)
        .await
        .with_context(|| format!("truncated payload in step {} reply", reply.step))?;
    Ok(StepOutput { reply, floats })
}

/// Server: read and check the client preface.
pub async fn read_preface<R: AsyncRead + Unpin>(r: &mut R) -> anyhow::Result<()> {
    let mut preface = [0u8; CLIENT_PREFACE.len()];
    r.read_exact(&mut preface)
        .await
        .context("missing shard client preface")?;
    ensure!(&preface == CLIENT_PREFACE, "invalid shard client preface");
    Ok(())
}

/// Server: send the startup metadata line and flush it.
pub async fn write_metadata<W: AsyncWrite + Unpin>(
    w: &mut W,
    meta: &StageMetadata,
) -> anyhow::Result<()> {
    let line = json_line(meta).context("failed to encode shard metadata")?;
    w.write_all(&line)
        .await
        .context("failed to send shard metadata")?;
    w.flush().await.context("failed to flush shard metadata")
}

/// Server: read the next request header, or `None` if the stream ends cleanly before one.
///
/// The magic, opcode and token count are checked here; step and position ordering are
/// session state that the caller checks.
pub async fn read_request_header<R: AsyncRead + Unpin>(
    r: &mut R,
) -> anyhow::Result<Option<RequestHeader>> {
    let mut bytes = [0u8; REQUEST_HEADER_LEN];
    let mut filled = 0;
    while filled < bytes.len() {
        let read = r
            .read(&mut bytes[filled..])
            .await
            .context("failed to read shard request header")?;
        if read == 0 {
            if filled == 0 {
                return Ok(None);
            }
            bail!("truncated shard request header ({filled} of {REQUEST_HEADER_LEN} bytes)");
        }
        filled += read;
    }
    parse_request_header(&bytes).map(Some)
}

/// Server: read `count` big-endian token IDs.
pub async fn read_tokens<R: AsyncRead + Unpin>(r: &mut R, count: u32) -> anyhow::Result<Vec<u32>> {
    check_token_count(count)?;
    let mut bytes = vec![0u8; count as usize * WORD_LEN];
    r.read_exact(&mut bytes)
        .await
        .context("truncated token payload")?;
    let (words, _) = bytes.as_chunks::<WORD_LEN>();
    Ok(words.iter().map(|&word| u32::from_be_bytes(word)).collect())
}

/// Server: read `values` little-endian hidden values (`token_count * embd`).
pub async fn read_hidden<R: AsyncRead + Unpin>(
    r: &mut R,
    values: usize,
) -> anyhow::Result<Vec<f32>> {
    ensure!(
        values <= MAX_HIDDEN_VALUES,
        "{values} hidden values exceed the bound of {MAX_HIDDEN_VALUES}"
    );
    read_f32s(r, values)
        .await
        .context("truncated hidden payload")
}

/// Server: send one reply line followed by its values, then flush.
pub async fn write_step<W: AsyncWrite + Unpin>(
    w: &mut W,
    reply: &StepReply,
    floats: &[f32],
) -> anyhow::Result<()> {
    ensure!(
        reply.n_floats == floats.len(),
        "step {} reply declares {} floats but carries {}",
        reply.step,
        reply.n_floats,
        floats.len()
    );
    let mut frame = json_line(reply).context("failed to encode shard step reply")?;
    frame.extend_from_slice(&encode_f32s(floats));
    w.write_all(&frame)
        .await
        .with_context(|| format!("failed to send step {} reply", reply.step))?;
    w.flush()
        .await
        .with_context(|| format!("failed to flush step {} reply", reply.step))
}

/// Encodes bytes as lowercase hex, the way the runner fills `piece_hex`.
pub fn encode_hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .flat_map(|byte| {
            [
                HEX_DIGITS[usize::from(byte >> 4)],
                HEX_DIGITS[usize::from(byte & 0x0f)],
            ]
        })
        .map(char::from)
        .collect()
}

fn check_token_count(count: u32) -> anyhow::Result<()> {
    ensure!(
        (1..=MAX_BATCH_TOKENS).contains(&count),
        "token count {count} outside 1..={MAX_BATCH_TOKENS}"
    );
    Ok(())
}

fn parse_request_header(bytes: &[u8; REQUEST_HEADER_LEN]) -> anyhow::Result<RequestHeader> {
    let mut words = [0u32; HEADER_WORDS];
    let (chunks, _) = bytes.as_chunks::<WORD_LEN>();
    for (word, &chunk) in words.iter_mut().zip(chunks) {
        *word = u32::from_be_bytes(chunk);
    }
    let [magic, opcode, step, position, token_count] = words;
    ensure!(
        magic == REQUEST_MAGIC,
        "bad shard request magic {magic:#010x}"
    );
    let opcode = Opcode::try_from(opcode)?;
    check_token_count(token_count)?;
    Ok(RequestHeader {
        opcode,
        step,
        position,
        token_count,
    })
}

fn payload_len(input: &StageInput<'_>) -> usize {
    match input {
        StageInput::Tokens(tokens) => tokens.len() * WORD_LEN,
        StageInput::Hidden { rows, .. } => rows.len() * WORD_LEN,
    }
}

fn encode_f32s(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Reads exactly `count` little-endian `f32` values; the caller has already bounded `count`.
async fn read_f32s<R: AsyncRead + Unpin>(r: &mut R, count: usize) -> anyhow::Result<Vec<f32>> {
    let len = count
        .checked_mul(WORD_LEN)
        .context("float payload length overflows")?;
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes).await?;
    let (words, _) = bytes.as_chunks::<WORD_LEN>();
    Ok(words.iter().map(|&word| f32::from_le_bytes(word)).collect())
}

/// Reads one newline-terminated JSON line of at most [`MAX_LINE_BYTES`] and decodes it.
async fn read_json_line<R, T>(r: &mut R, what: &str) -> anyhow::Result<T>
where
    R: AsyncBufRead + Unpin,
    T: DeserializeOwned,
{
    let mut line = Vec::new();
    let limit = MAX_LINE_BYTES as u64 + 1; // the content plus its newline
    (&mut *r)
        .take(limit)
        .read_until(b'\n', &mut line)
        .await
        .with_context(|| format!("failed to read {what} line"))?;
    if line.last() != Some(&b'\n') {
        if line.len() > MAX_LINE_BYTES {
            bail!("{what} line exceeds {MAX_LINE_BYTES} bytes");
        }
        let place = if line.is_empty() {
            "before"
        } else {
            "in the middle of"
        };
        return Err(StreamClosed {
            place,
            what: what.to_owned(),
        }
        .into());
    }
    serde_json::from_slice(&line).with_context(|| format!("invalid {what} JSON"))
}

fn json_line<T: Serialize>(value: &T) -> anyhow::Result<Vec<u8>> {
    let mut line = serde_json::to_vec(value)?;
    line.push(b'\n');
    Ok(line)
}

fn decode_hex(hex: &str) -> anyhow::Result<Vec<u8>> {
    let digits = hex.as_bytes();
    let (pairs, odd) = digits.as_chunks::<2>();
    ensure!(
        odd.is_empty(),
        "odd-length hex string ({} digits)",
        digits.len()
    );
    pairs
        .iter()
        .map(|&[high, low]| Ok((hex_value(high)? << 4) | hex_value(low)?))
        .collect()
}

fn hex_value(digit: u8) -> anyhow::Result<u8> {
    let value = char::from(digit)
        .to_digit(16)
        .with_context(|| format!("invalid hex digit {:?}", char::from(digit)))?;
    Ok(value as u8)
}

#[cfg(test)]
mod tests {
    use tokio::io::{BufReader, BufWriter, duplex};

    use super::*;

    /// A final-stage reply exactly as the C++ runner's printf emits it on Linux.
    const FINAL_REPLY_LINE: &str = "{\"step\":3,\"n_floats\":0,\"token\":9707,\"eog\":false,\
        \"piece_hex\":\"48656c6c6f\",\"compute_ms\":12.345678,\
        \"observed_peak_buffer_bytes\":123456,\"max_rss_bytes\":987654,\"vm\":null}\n";

    /// A startup line as the C++ runner prints it on macOS (with VM metrics).
    const MAC_METADATA_LINE: &str = "{\"protocol\":1,\"begin\":0,\"end\":32,\"layers\":64,\
        \"embd\":5120,\"vocab\":152064,\"allocated_bytes\":1048576,\"max_rss_bytes\":2097152,\
        \"vm\":{\"resident_bytes\":1,\"physical_footprint_bytes\":2,\
        \"physical_footprint_peak_bytes\":3,\"compressed_bytes\":4},\
        \"source_sha256\":\"\",\"tokens\":[151644,872,198]}\n";

    fn header_bytes(words: [u32; HEADER_WORDS]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_be_bytes()).collect()
    }

    fn reply(step: u32, n_floats: usize, piece_hex: &str) -> StepReply {
        StepReply {
            step,
            n_floats,
            token: -1,
            eog: false,
            piece_hex: piece_hex.to_string(),
            compute_ms: 1.5,
            observed_peak_buffer_bytes: 64,
            extra: Map::new(),
        }
    }

    #[tokio::test]
    async fn token_request_has_golden_bytes_and_is_flushed() {
        let mut writer = BufWriter::new(Vec::new());
        write_request(&mut writer, 0, 0, &StageInput::Tokens(&[1, 2, 0x0102_0304]))
            .await
            .unwrap();
        let mut expected = header_bytes([0x4d53_4831, 1, 0, 0, 3]);
        expected.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 2, 1, 2, 3, 4]);
        assert_eq!(writer.get_ref(), &expected);
        assert_eq!(&writer.get_ref()[..4], b"MSH1");
    }

    #[tokio::test]
    async fn hidden_request_has_golden_bytes() {
        let mut out = Vec::new();
        let rows = [1.0f32, -2.5, 0.0, 4.0];
        write_request(
            &mut out,
            7,
            40,
            &StageInput::Hidden {
                rows: &rows,
                embd: 2,
            },
        )
        .await
        .unwrap();
        let mut expected = header_bytes([REQUEST_MAGIC, 2, 7, 40, 2]);
        for value in rows {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(out, expected);
    }

    #[tokio::test]
    async fn requests_round_trip_over_a_duplex() {
        let (mut client, mut server) = duplex(1024);
        let rows = [0.5f32, 1.5, 2.5];
        write_request(&mut client, 0, 0, &StageInput::Tokens(&[5, 6]))
            .await
            .unwrap();
        write_request(
            &mut client,
            1,
            2,
            &StageInput::Hidden {
                rows: &rows,
                embd: 3,
            },
        )
        .await
        .unwrap();
        drop(client);

        let first = read_request_header(&mut server).await.unwrap().unwrap();
        let expected = RequestHeader {
            opcode: Opcode::Tokens,
            step: 0,
            position: 0,
            token_count: 2,
        };
        assert_eq!(first, expected);
        assert_eq!(read_tokens(&mut server, 2).await.unwrap(), vec![5, 6]);

        let second = read_request_header(&mut server).await.unwrap().unwrap();
        assert_eq!(second.opcode, Opcode::Hidden);
        assert_eq!(
            (second.step, second.position, second.token_count),
            (1, 2, 1)
        );
        assert_eq!(read_hidden(&mut server, 3).await.unwrap(), rows.to_vec());
        assert!(read_request_header(&mut server).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn preface_round_trips_and_rejects_garbage() {
        let mut out = Vec::new();
        write_preface(&mut out).await.unwrap();
        assert_eq!(out, b"MSH1\n");
        read_preface(&mut out.as_slice()).await.unwrap();
        assert!(read_preface(&mut &b"MSH2\n"[..]).await.is_err());
        assert!(read_preface(&mut &b"MS"[..]).await.is_err());
    }

    #[tokio::test]
    async fn parses_linux_metadata_and_keeps_extra_fields() {
        let line = "{\"protocol\":1,\"begin\":32,\"end\":64,\"layers\":64,\"embd\":5120,\
            \"vocab\":152064,\"allocated_bytes\":42,\"max_rss_bytes\":99,\"vm\":null,\
            \"source_sha256\":\"\",\"tokens\":[1,2,3],\"future_field\":\"x\"}\n";
        let meta = read_metadata(&mut line.as_bytes()).await.unwrap();
        assert_eq!((meta.begin, meta.end, meta.layers), (32, 64, 64));
        assert_eq!(
            (meta.embd, meta.vocab, meta.allocated_bytes),
            (5120, 152064, 42)
        );
        assert_eq!(meta.tokens, vec![1, 2, 3]);
        assert!(meta.is_final());
        assert_eq!(meta.model_sha256, None);
        assert_eq!(meta.extra.get("vm"), Some(&Value::Null));
        assert_eq!(meta.extra.get("max_rss_bytes"), Some(&Value::from(99)));
        assert_eq!(meta.extra.get("future_field"), Some(&Value::from("x")));
    }

    #[tokio::test]
    async fn metadata_round_trips_with_vm_object_and_relay_fields() {
        let mut meta = read_metadata(&mut MAC_METADATA_LINE.as_bytes())
            .await
            .unwrap();
        assert!(!meta.is_final());
        assert_eq!(meta.extra["vm"]["compressed_bytes"], Value::from(4));
        meta.model_sha256 = Some("a".repeat(64));
        meta.artifact_sha256 = Some("b".repeat(64));
        meta.runner_sha256 = Some("c".repeat(64));

        let mut out = Vec::new();
        write_metadata(&mut out, &meta).await.unwrap();
        assert_eq!(out.last(), Some(&b'\n'));
        assert_eq!(out.iter().filter(|&&byte| byte == b'\n').count(), 1);
        let decoded = read_metadata(&mut out.as_slice()).await.unwrap();
        assert_eq!(decoded, meta);
    }

    #[tokio::test]
    async fn parses_final_reply_from_runner() {
        let output = read_step(&mut FINAL_REPLY_LINE.as_bytes(), 0)
            .await
            .unwrap();
        let reply = output.reply;
        assert_eq!((reply.step, reply.n_floats, reply.token), (3, 0, 9707));
        assert!(!reply.eog);
        assert_eq!(reply.piece_bytes().unwrap(), b"Hello");
        assert_eq!(reply.compute_ms, 12.345678);
        assert_eq!(reply.observed_peak_buffer_bytes, 123456);
        assert_eq!(reply.extra.get("vm"), Some(&Value::Null));
        assert_eq!(reply.extra.get("max_rss_bytes"), Some(&Value::from(987654)));
        assert!(output.floats.is_empty());
    }

    #[tokio::test]
    async fn parses_prefix_reply_with_floats() {
        let mut bytes =
            b"{\"step\":0,\"n_floats\":2,\"token\":-1,\"eog\":false,\"piece_hex\":\"\",\
            \"compute_ms\":0.500000,\"observed_peak_buffer_bytes\":1,\"max_rss_bytes\":2,\
            \"vm\":{\"resident_bytes\":5}}\n"
                .to_vec();
        bytes.extend_from_slice(&1.25f32.to_le_bytes());
        bytes.extend_from_slice(&(-3.0f32).to_le_bytes());
        let output = read_step(&mut bytes.as_slice(), 2).await.unwrap();
        assert_eq!(output.reply.token, -1);
        assert_eq!(output.floats, vec![1.25, -3.0]);
        assert_eq!(output.reply.extra["vm"]["resident_bytes"], Value::from(5));
    }

    #[tokio::test]
    async fn step_round_trips_over_a_duplex() {
        let (mut server, client) = duplex(1024);
        let mut client = BufReader::new(client);
        let mut first = reply(0, 3, "");
        first.extra.insert("vm".to_string(), Value::Null);
        let mut second = reply(1, 0, &encode_hex(&[0xe2, 0x82]));
        second.token = 42;
        second.eog = true;
        write_step(&mut server, &first, &[1.0, 2.0, 3.0])
            .await
            .unwrap();
        write_step(&mut server, &second, &[]).await.unwrap();

        let output = read_step(&mut client, 3).await.unwrap();
        assert_eq!(output.reply, first);
        assert_eq!(output.floats, vec![1.0, 2.0, 3.0]);
        let output = read_step(&mut client, 0).await.unwrap();
        assert_eq!(output.reply, second);
        assert_eq!(output.reply.piece_bytes().unwrap(), vec![0xe2, 0x82]);
    }

    #[test]
    fn hex_round_trips_every_byte() {
        let bytes: Vec<u8> = (0..=u8::MAX).collect();
        let hex = encode_hex(&bytes);
        assert_eq!(&hex[..6], "000102");
        assert_eq!(decode_hex(&hex).unwrap(), bytes);
        assert_eq!(decode_hex("ABcd").unwrap(), vec![0xab, 0xcd]);
    }

    #[tokio::test]
    async fn rejects_oversize_line() {
        let mut line = vec![b' '; MAX_LINE_BYTES + 1];
        line.push(b'\n');
        let error = read_metadata(&mut line.as_slice()).await.unwrap_err();
        assert!(error.to_string().contains("exceeds"), "{error:#}");
    }

    #[tokio::test]
    async fn accepts_line_at_the_bound() {
        let json = b"{\"step\":0,\"n_floats\":0,\"token\":1,\"eog\":true}";
        let mut line = json.to_vec();
        line.resize(MAX_LINE_BYTES, b' ');
        line.push(b'\n');
        let output = read_step(&mut line.as_slice(), 0).await.unwrap();
        assert!(output.reply.eog);
    }

    #[tokio::test]
    async fn rejects_missing_or_truncated_line() {
        assert!(read_metadata(&mut &b""[..]).await.is_err());
        assert!(read_metadata(&mut &b"{\"protocol\":1"[..]).await.is_err());
        assert!(read_metadata(&mut &b"not json\n"[..]).await.is_err());
    }

    #[tokio::test]
    async fn rejects_bad_magic() {
        let bytes = header_bytes([0x4d53_4832, 1, 0, 0, 1]);
        let error = read_request_header(&mut bytes.as_slice())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("magic"), "{error:#}");
    }

    #[tokio::test]
    async fn rejects_unknown_opcode() {
        let bytes = header_bytes([REQUEST_MAGIC, 3, 0, 0, 1]);
        assert!(read_request_header(&mut bytes.as_slice()).await.is_err());
    }

    #[tokio::test]
    async fn rejects_token_count_out_of_range() {
        for count in [0, MAX_BATCH_TOKENS + 1] {
            let bytes = header_bytes([REQUEST_MAGIC, 1, 0, 0, count]);
            assert!(read_request_header(&mut bytes.as_slice()).await.is_err());
            assert!(read_tokens(&mut &[0u8; 1024][..], count).await.is_err());
        }
    }

    #[tokio::test]
    async fn rejects_truncated_header() {
        let bytes = header_bytes([REQUEST_MAGIC, 1, 0, 0, 1]);
        let error = read_request_header(&mut &bytes[..7]).await.unwrap_err();
        assert!(error.to_string().contains("truncated"), "{error:#}");
    }

    #[tokio::test]
    async fn write_request_rejects_bad_batches() {
        let too_many = vec![0u32; MAX_BATCH_TOKENS as usize + 1];
        let cases = [
            StageInput::Tokens(&[]),
            StageInput::Tokens(&too_many),
            StageInput::Hidden {
                rows: &[1.0, 2.0, 3.0],
                embd: 2,
            },
            StageInput::Hidden {
                rows: &[1.0],
                embd: 0,
            },
            StageInput::Hidden { rows: &[], embd: 4 },
        ];
        for input in cases {
            let mut out = Vec::new();
            assert!(write_request(&mut out, 0, 0, &input).await.is_err());
            assert!(out.is_empty());
        }
    }

    #[tokio::test]
    async fn rejects_n_floats_over_the_bound() {
        let mut bytes = Vec::new();
        write_step(&mut bytes, &reply(0, 4, ""), &[0.0; 4])
            .await
            .unwrap();
        let error = read_step(&mut bytes.as_slice(), 3).await.err().unwrap();
        assert!(error.to_string().contains("more than"), "{error:#}");
        assert!(
            read_hidden(&mut &[0u8; 4][..], MAX_HIDDEN_VALUES + 1)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn rejects_bad_piece_hex() {
        for piece_hex in ["abc", "zz", "4g"] {
            let mut bytes = Vec::new();
            write_step(&mut bytes, &reply(0, 0, piece_hex), &[])
                .await
                .unwrap();
            assert!(
                read_step(&mut bytes.as_slice(), 0).await.is_err(),
                "{piece_hex}"
            );
        }
    }

    #[tokio::test]
    async fn rejects_truncated_payloads() {
        let mut bytes = Vec::new();
        write_step(&mut bytes, &reply(0, 2, ""), &[1.0, 2.0])
            .await
            .unwrap();
        bytes.truncate(bytes.len() - 1);
        let error = read_step(&mut bytes.as_slice(), 2).await.err().unwrap();
        assert!(error.to_string().contains("truncated"), "{error:#}");
        assert!(read_tokens(&mut &[0u8; 7][..], 2).await.is_err());
        assert!(read_hidden(&mut &[0u8; 7][..], 2).await.is_err());
    }

    #[tokio::test]
    async fn write_step_rejects_mismatched_float_count() {
        let mut out = Vec::new();
        assert!(
            write_step(&mut out, &reply(0, 2, ""), &[1.0])
                .await
                .is_err()
        );
        assert!(out.is_empty());
    }
}
