//! Wire protocol: versioned messages, serialization, and stream framing.
//!
//! Every control-stream message is a [`Message`] encoded with postcard and sent as one frame:
//! a big-endian `u32` length prefix followed by that many payload bytes.
//!
//! Postcard identifies enum variants by their position, so new [`Message`] variants must be
//! appended at the end and existing fields must not be reordered without bumping
//! [`PROTOCOL_VERSION`].

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Version of the message schema; peers exchange it in [`Hello`].
pub const PROTOCOL_VERSION: u16 = 1;

/// ALPN for the control connection that carries framed [`Message`]s.
pub const CONTROL_ALPN: &[u8] = b"meshthing/control/1";

/// ALPN for the raw-byte bridge that tunnels the inference RPC between leader and worker.
pub const RPC_BRIDGE_ALPN: &[u8] = b"meshthing/rpc-bridge/1";

/// Largest frame payload accepted on read or produced on write, in bytes.
pub const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

/// Size of the big-endian length prefix that starts every frame.
const LENGTH_PREFIX_LEN: usize = std::mem::size_of::<u32>();

/// Every message that can travel on a control stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Message {
    /// First message on a new connection, sent by both sides.
    Hello(Hello),
    /// Periodic liveness signal.
    Heartbeat(Heartbeat),
    /// A user prompt to run through the model.
    Prompt(Prompt),
    /// Intermediate activations passed between pipeline stages.
    Activations(Activations),
    /// Conversation history synchronisation.
    History(History),
}

/// Handshake identifying a node and how far its log has progressed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hello {
    /// Sender's Iroh node ID as text.
    pub node_id: String,
    /// Sender's [`PROTOCOL_VERSION`].
    pub protocol_version: u16,
    /// Cluster the sender belongs to; peers from other clusters are rejected.
    pub cluster_id: String,
    /// Index of the last entry in the sender's replicated log.
    pub last_log_index: u64,
}

/// Liveness signal from a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Heartbeat {
    /// Sender's Iroh node ID as text.
    pub node_id: String,
    /// Leadership epoch the sender believes is current.
    pub epoch: u64,
    /// Monotonic counter within the epoch, used to spot gaps and reordering.
    pub sequence: u64,
}

/// A prompt submitted for inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prompt {
    /// Identifier that ties activations and responses back to this prompt.
    pub request_id: String,
    /// Conversation the prompt belongs to.
    pub conversation_id: String,
    /// Prompt text.
    pub text: String,
}

/// Activations produced by one pipeline stage for the next.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Activations {
    /// Prompt these activations were computed for.
    pub request_id: String,
    /// Model layer that produced the activations.
    pub layer: u32,
    /// Serialized tensor data.
    pub data: Vec<u8>,
}

/// Conversation history being synchronised between nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct History {
    /// Conversation the history belongs to.
    pub conversation_id: String,
    /// Opaque payload; its format is owned by the history-sync component.
    pub payload: Vec<u8>,
}

/// Errors from encoding, decoding, or framing messages.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    /// The underlying stream failed, including EOF in the middle of a frame.
    #[error("stream i/o failed: {0}")]
    Io(#[from] std::io::Error),
    /// A message could not be serialized.
    #[error("failed to encode message: {0}")]
    Encode(#[source] postcard::Error),
    /// Received bytes are not a valid message.
    #[error("failed to decode message: {0}")]
    Decode(#[source] postcard::Error),
    /// A frame exceeded the limit for its channel.
    #[error("frame of {len} bytes exceeds the {limit}-byte limit")]
    FrameTooLarge {
        /// Length of the rejected frame payload in bytes.
        len: usize,
        /// Maximum accepted frame payload in bytes.
        limit: usize,
    },
}

/// Serializes a message to postcard bytes (without a frame header).
pub fn encode(message: &Message) -> Result<Vec<u8>, ProtocolError> {
    postcard::to_stdvec(message).map_err(ProtocolError::Encode)
}

/// Deserializes a message from postcard bytes (without a frame header).
pub fn decode(bytes: &[u8]) -> Result<Message, ProtocolError> {
    postcard::from_bytes(bytes).map_err(ProtocolError::Decode)
}

/// Writes one frame: a `u32` big-endian length prefix followed by the encoded message.
///
/// Rejects messages whose encoding is larger than [`MAX_FRAME_LEN`] without allocating the
/// encoded payload or writing anything.
pub async fn write_frame<W: AsyncWrite + Unpin>(
    writer: &mut W,
    message: &Message,
) -> Result<(), ProtocolError> {
    // Measure first: `encode` allocates a `Vec`, so doing this after encoding would let an
    // oversized activation/history payload allocate far beyond the transport's frame limit.
    let encoded_len =
        postcard::experimental::serialized_size(message).map_err(ProtocolError::Encode)?;
    if encoded_len > MAX_FRAME_LEN {
        return Err(ProtocolError::FrameTooLarge {
            len: encoded_len,
            limit: MAX_FRAME_LEN,
        });
    }
    let payload = encode(message)?;
    let len = payload.len();
    debug_assert_eq!(len, encoded_len);
    let prefix = u32::try_from(len).map_err(|_| ProtocolError::FrameTooLarge {
        len,
        limit: MAX_FRAME_LEN,
    })?;
    writer.write_all(&prefix.to_be_bytes()).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

/// Reads one frame and decodes it.
///
/// Returns `Ok(None)` if the stream ends cleanly before a length prefix starts. EOF partway
/// through a frame is an [`ProtocolError::Io`] error. An oversized length prefix is rejected
/// with [`ProtocolError::FrameTooLarge`] before any payload buffer is allocated.
pub async fn read_frame<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<Option<Message>, ProtocolError> {
    read_frame_with_limit(reader, MAX_FRAME_LEN).await
}

/// Reads one frame and decodes it, enforcing a caller-specific payload bound.
///
/// This is useful for protocol channels whose messages are smaller than the general frame limit.
/// Like [`read_frame`], it rejects an oversized length prefix before allocating its payload.
pub async fn read_frame_with_limit<R: AsyncRead + Unpin>(
    reader: &mut R,
    max_len: usize,
) -> Result<Option<Message>, ProtocolError> {
    let Some(len) = read_length_prefix(reader).await? else {
        return Ok(None);
    };
    if len > max_len {
        return Err(ProtocolError::FrameTooLarge {
            len,
            limit: max_len,
        });
    }
    let mut payload = vec![0; len];
    reader.read_exact(&mut payload).await?;
    decode(&payload).map(Some)
}

/// Reads the frame length prefix, distinguishing a clean EOF (`None`) from a truncated prefix.
async fn read_length_prefix<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<Option<usize>, ProtocolError> {
    let mut prefix = [0; LENGTH_PREFIX_LEN];
    let mut filled = 0;
    while filled < LENGTH_PREFIX_LEN {
        let read = reader.read(&mut prefix[filled..]).await?;
        if read == 0 {
            if filled == 0 {
                return Ok(None);
            }
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }
        filled += read;
    }
    let len = u32::from_be_bytes(prefix);
    let len = usize::try_from(len).map_err(|_| ProtocolError::FrameTooLarge {
        len: usize::MAX,
        limit: usize::MAX,
    })?;
    Ok(Some(len))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Buffer size for in-memory duplex streams; large enough for every test frame.
    const DUPLEX_CAPACITY: usize = 64 * 1024;

    fn sample_messages() -> Vec<Message> {
        vec![
            Message::Hello(Hello {
                node_id: "node-a".into(),
                protocol_version: PROTOCOL_VERSION,
                cluster_id: "cluster-1".into(),
                last_log_index: 42,
            }),
            Message::Heartbeat(Heartbeat {
                node_id: "node-b".into(),
                epoch: 3,
                sequence: u64::MAX,
            }),
            Message::Prompt(Prompt {
                request_id: "req-1".into(),
                conversation_id: "conv-1".into(),
                text: "Hello, mesh! \u{1F310}".into(),
            }),
            Message::Activations(Activations {
                request_id: "req-1".into(),
                layer: 17,
                data: (0..=255).collect(),
            }),
            Message::History(History {
                conversation_id: "conv-1".into(),
                payload: vec![],
            }),
        ]
    }

    #[test]
    fn every_variant_round_trips_through_encode_and_decode() {
        for message in sample_messages() {
            let bytes = encode(&message).unwrap();
            assert_eq!(decode(&bytes).unwrap(), message);
        }
    }

    #[test]
    fn decode_rejects_garbage() {
        assert!(matches!(decode(&[0xFF]), Err(ProtocolError::Decode(_))));
    }

    #[tokio::test]
    async fn every_variant_round_trips_through_frames() {
        for message in sample_messages() {
            let (mut writer, mut reader) = tokio::io::duplex(DUPLEX_CAPACITY);
            write_frame(&mut writer, &message).await.unwrap();
            assert_eq!(read_frame(&mut reader).await.unwrap(), Some(message));
        }
    }

    #[tokio::test]
    async fn back_to_back_frames_then_clean_eof() {
        let messages = sample_messages();
        let (mut writer, mut reader) = tokio::io::duplex(DUPLEX_CAPACITY);
        for message in &messages {
            write_frame(&mut writer, message).await.unwrap();
        }
        drop(writer);
        for message in messages {
            assert_eq!(read_frame(&mut reader).await.unwrap(), Some(message));
        }
        assert_eq!(read_frame(&mut reader).await.unwrap(), None);
    }

    #[tokio::test]
    async fn empty_stream_is_clean_eof() {
        let mut reader: &[u8] = &[];
        assert_eq!(read_frame(&mut reader).await.unwrap(), None);
    }

    #[tokio::test]
    async fn truncated_length_prefix_is_an_error() {
        let mut reader: &[u8] = &[0, 0];
        let error = read_frame(&mut reader).await.unwrap_err();
        assert!(is_unexpected_eof(&error), "unexpected error: {error}");
    }

    #[tokio::test]
    async fn truncated_payload_is_an_error() {
        let mut frame = Vec::new();
        write_frame(&mut frame, &sample_messages()[0])
            .await
            .unwrap();
        frame.pop();
        let mut reader = frame.as_slice();
        let error = read_frame(&mut reader).await.unwrap_err();
        assert!(is_unexpected_eof(&error), "unexpected error: {error}");
    }

    #[tokio::test]
    async fn oversized_length_prefix_is_rejected_before_reading_payload() {
        // No payload follows the prefix: if read_frame tried to allocate and read the
        // payload it would fail with EOF instead of FrameTooLarge.
        for len in [MAX_FRAME_LEN + 1, u32::MAX as usize] {
            let prefix = u32::try_from(len).unwrap().to_be_bytes();
            let mut reader = prefix.as_slice();
            let error = read_frame(&mut reader).await.unwrap_err();
            assert!(
                matches!(error, ProtocolError::FrameTooLarge { len: rejected, limit: MAX_FRAME_LEN } if rejected == len),
                "unexpected error: {error}"
            );
        }
    }

    #[tokio::test]
    async fn caller_frame_limit_is_checked_before_reading_payload() {
        let prefix = 1025u32.to_be_bytes();
        let mut reader = prefix.as_slice();
        let error = read_frame_with_limit(&mut reader, 1024).await.unwrap_err();
        assert!(matches!(
            error,
            ProtocolError::FrameTooLarge {
                len: 1025,
                limit: 1024
            }
        ));
    }

    #[tokio::test]
    async fn oversized_message_is_rejected_without_writing() {
        let message = Message::Activations(Activations {
            request_id: "req-big".into(),
            layer: 0,
            data: vec![0; MAX_FRAME_LEN],
        });
        let mut written = Vec::new();
        let error = write_frame(&mut written, &message).await.unwrap_err();
        let encoded_len = postcard::experimental::serialized_size(&message).unwrap();
        assert!(
            matches!(error, ProtocolError::FrameTooLarge { len, limit: MAX_FRAME_LEN } if len == encoded_len && len > MAX_FRAME_LEN),
            "unexpected error: {error}"
        );
        assert!(written.is_empty());
    }

    fn is_unexpected_eof(error: &ProtocolError) -> bool {
        matches!(error, ProtocolError::Io(io) if io.kind() == std::io::ErrorKind::UnexpectedEof)
    }
}
