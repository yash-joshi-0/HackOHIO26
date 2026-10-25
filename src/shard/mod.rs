//! Layer-sharded execution backend built on david-codex's v1 shard runner.
//!
//! A model is split into contiguous layer ranges ("stages"), each run by a `shard_runner`
//! process speaking a small line-plus-binary protocol over stdio ([`wire`]). Remote stages are
//! relayed over Iroh by [`serve`]; the leader validates a [`manifest`], drives the stages in
//! order with [`pipeline`], and exposes the result as an OpenAI-shaped HTTP API in [`frontend`].
//! A balanced manifest chooses the ranges afresh at every prompt start with [`balance`].

pub mod balance;
pub mod frontend;
pub mod manifest;
pub mod pipeline;
pub mod ring;
pub mod ring_origin;
pub mod serve;
pub mod template;
pub mod wire;
