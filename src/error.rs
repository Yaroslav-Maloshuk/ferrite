use thiserror::Error;

#[derive(Debug, Error)]
pub enum FerriteError {
    #[error("model error: {0}")]
    Model(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("database error: {0}")]
    Database(String),
    #[error("http error: {0}")]
    Http(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error("{0}")]
    Other(String),
}
