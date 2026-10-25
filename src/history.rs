//! Hooks for a job's conversation turn, consumed by history storage and sync.
//!
//! The API reports every job that reaches the run slot to a [`HistorySink`]: `started` when it
//! is dequeued, then either `record` with the completed turn or `ended` with its terminal outcome
//! (failure, cancellation, or successful completion without history retention). Only a logging
//! sink ships here; persistent storage and replication plug in behind the same trait (M-9, M-10).

use serde_json::Value;
use tracing::info;

/// Top-level request field holding the chat messages, counted (never logged) by the logging sink.
const MESSAGES_FIELD: &str = "messages";

/// A job that has just been given the run slot and is about to be forwarded upstream.
#[derive(Clone, Debug)]
pub struct StartedTurn {
    /// The job's UUID, matching `JobInfo::request_id`.
    pub request_id: String,
    /// Conversation the client named in its request, if any.
    pub conversation_id: Option<String>,
    /// Chat-completions request body as forwarded (`conversation_id` removed), as in
    /// [`CompletedTurn::request`].
    pub request: Value,
}

/// One finished request/reply pair, handed to a [`HistorySink`] after its job completes.
#[derive(Clone, Debug)]
pub struct CompletedTurn {
    /// The job's UUID, matching `JobInfo::request_id`.
    pub request_id: String,
    /// Conversation the client named in its request, if any.
    pub conversation_id: Option<String>,
    /// Chat-completions request body as forwarded to llama-server (`conversation_id` removed).
    pub request: Value,
    /// Assistant reply text; assembled from the delta chunks for streamed replies.
    pub response_text: String,
}

/// A started job's terminal outcome when its reply is not retained in history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TurnOutcome {
    /// Upstream error, non-2xx reply, or an invalid or truncated reply.
    Failed {
        /// Same text as `JobInfo::error`.
        reason: String,
    },
    /// The client went away while the job was running.
    Canceled,
    /// The request succeeded, but its assistant response exceeded the history retention limit.
    CompletedWithoutHistory {
        /// Maximum assistant response size retained by the API.
        limit_bytes: usize,
    },
}

/// Receives a job's lifecycle. For every job that acquires the run slot, [`started`] is called
/// exactly once, followed by exactly one of [`record`] or [`ended`]. Jobs canceled while still
/// queued produce no calls. A job's last call happens before the next job can start, so calls
/// for different jobs never interleave.
///
/// Every method is called on the request's task, so implementations should return quickly and
/// move slow work (disk, network) onto their own task.
///
/// [`started`]: HistorySink::started
/// [`record`]: HistorySink::record
/// [`ended`]: HistorySink::ended
pub trait HistorySink: Send + Sync {
    /// Called once when the job acquires the run slot, before it is forwarded upstream.
    /// Default: no-op.
    fn started(&self, _turn: &StartedTurn) {}

    /// Takes ownership of one completed turn. Called once when the job completes with a valid
    /// reply, and never together with [`ended`](HistorySink::ended) for the same job.
    fn record(&self, turn: CompletedTurn);

    /// Called once when a started job fails, is canceled, or completes without history retention;
    /// never after [`record`](HistorySink::record). Default: no-op.
    fn ended(&self, _request_id: &str, _outcome: TurnOutcome) {}
}

/// Sink that only logs each lifecycle call via `tracing` (target `history`); nothing is stored.
/// Logs ids, counts and outcomes only, never prompt or reply content.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoggingHistorySink;

impl HistorySink for LoggingHistorySink {
    fn started(&self, turn: &StartedTurn) {
        let message_count = turn
            .request
            .get(MESSAGES_FIELD)
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        info!(
            target: "history",
            request_id = %turn.request_id,
            conversation_id = turn.conversation_id.as_deref().unwrap_or("-"),
            message_count,
            "started turn"
        );
    }

    fn record(&self, turn: CompletedTurn) {
        info!(
            target: "history",
            request_id = %turn.request_id,
            conversation_id = turn.conversation_id.as_deref().unwrap_or("-"),
            response_chars = turn.response_text.chars().count(),
            "completed turn"
        );
    }

    fn ended(&self, request_id: &str, outcome: TurnOutcome) {
        match outcome {
            TurnOutcome::Failed { reason } => {
                info!(target: "history", request_id, %reason, "failed turn");
            }
            TurnOutcome::Canceled => info!(target: "history", request_id, "canceled turn"),
            TurnOutcome::CompletedWithoutHistory { limit_bytes } => info!(
                target: "history",
                request_id,
                limit_bytes,
                "completed turn omitted from history"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Implements only the required method, so the lifecycle hooks must have defaults.
    struct RecordOnlySink;

    impl HistorySink for RecordOnlySink {
        fn record(&self, _turn: CompletedTurn) {}
    }

    fn started_turn() -> StartedTurn {
        StartedTurn {
            request_id: "id".to_owned(),
            conversation_id: Some("conv".to_owned()),
            request: serde_json::json!({"messages": [{"role": "user", "content": "hi"}]}),
        }
    }

    fn completed_turn() -> CompletedTurn {
        CompletedTurn {
            request_id: "id".to_owned(),
            conversation_id: None,
            request: serde_json::json!({"messages": []}),
            response_text: "hello".to_owned(),
        }
    }

    #[test]
    fn logging_sink_accepts_every_lifecycle_call_behind_a_trait_object() {
        let sink: Box<dyn HistorySink> = Box::new(LoggingHistorySink);
        sink.started(&started_turn());
        sink.record(completed_turn());
        sink.ended(
            "id",
            TurnOutcome::Failed {
                reason: "upstream returned 500".to_owned(),
            },
        );
        sink.ended("id", TurnOutcome::Canceled);
    }

    #[test]
    fn record_only_sinks_get_no_op_lifecycle_hooks() {
        let sink: Box<dyn HistorySink> = Box::new(RecordOnlySink);
        sink.started(&started_turn());
        sink.ended("id", TurnOutcome::Canceled);
        sink.record(completed_turn());
    }
}
