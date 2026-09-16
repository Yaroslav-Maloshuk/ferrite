pub mod config;
pub mod embedding;
pub mod error;
pub mod http;
pub mod pipeline;
pub mod pooling;
pub mod store;

pub use config::{FerriteConfig, IndexMode};
pub use error::FerriteError;
#[cfg(feature = "bench")]
pub mod bench;

pub use pipeline::{Ferrite, IngestItem};
