pub(crate) mod hash_iterators;
pub(crate) mod utils;
pub mod metrics;

//re-exports
pub use metrics::rouge_n::rouge_n_similarity;
pub use metrics::jaccard::jaccard_n_similarity;
pub use metrics::bleu::bleu_similarity;
pub use metrics::cosine::{cosine_similarity, CosineSimilarityError};
