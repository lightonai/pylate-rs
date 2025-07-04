use serde::{Deserialize, Serialize};

/// Input structure for similarity computation.
///
/// Contains two lists of strings: queries and documents, for which
/// a similarity matrix will be computed.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SimilarityInput {
    /// A list of query strings.
    pub queries: Vec<String>,
    /// A list of document strings.
    pub documents: Vec<String>,
}

/// Input structure for the encoding process.
///
/// Contains a single list of sentences to be encoded into embeddings.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EncodeInput {
    /// A list of sentences (queries or documents) to be encoded.
    pub sentences: Vec<String>,
    /// An optional batch size to override the model's default.
    pub batch_size: Option<usize>,
}

/// Output structure for the encoding process.
///
/// Contains the resulting embeddings for a batch of sentences.
#[derive(Serialize, Deserialize, Debug)]
pub struct EncodeOutput {
    /// A nested vector representing the embeddings.
    /// The structure is `[batch_size, sequence_length, embedding_dimension]`.
    pub embeddings: Vec<Vec<Vec<f32>>>,
}

/// Output structure for the similarity computation.
///
/// Contains a matrix of similarity scores between queries and documents.
#[derive(Serialize, Deserialize, Debug)]
pub struct Similarities {
    /// A 2D vector where `data[i][j]` is the similarity score
    /// between the i-th query and the j-th document.
    pub data: Vec<Vec<f32>>,
}

/// Output structure for the raw similarity matrix computation.
///
/// This provides a detailed, un-reduced view of the similarity scores,
/// along with the tokens for queries and documents for inspection.
#[derive(Serialize, Deserialize, Debug)]
pub struct RawSimilarityOutput {
    /// The raw similarity matrix with dimensions
    /// `[num_queries, num_documents, query_length, document_length]`.
    pub similarity_matrix: Vec<Vec<Vec<Vec<f32>>>>,
    /// The tokens corresponding to each query.
    pub query_tokens: Vec<Vec<String>>,
    /// The tokens corresponding to each document.
    pub document_tokens: Vec<Vec<String>>,
}
