#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod builder;
pub mod error;
pub mod model;
pub mod pooling;
pub mod types;
pub mod utils;
#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use builder::ColbertBuilder;
pub use error::ColbertError;
pub use model::{BaseModel, ColBERT};
pub use pooling::hierarchical_pooling;
pub use types::{EncodeInput, EncodeOutput, RawSimilarityOutput, Similarities, SimilarityInput};
pub use utils::normalize_l2;

#[cfg(feature = "python")]
pub mod python;
