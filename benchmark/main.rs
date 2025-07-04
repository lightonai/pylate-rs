use anyhow::Result;
use candle_core::Device;
use pylate_rs::{hierarchical_pooling, ColBERT};

fn main() -> Result<()> {
    let device = Device::Cpu;
    // let device = Device::new_cuda(0)?; // Uncomment this line to use GPU if available
    // let device = Device::new_metal(0)?; // Uncomment this line to use Apple Silicon GPU if available

    let mut model: ColBERT = ColBERT::from("lightonai/colbertv2.0")
        .with_device(device)
        .try_into()?;

    let query_sentences = vec!["Query 1.".to_string(), "Query 2.".to_string()];
    let document_sentences = vec!["Document 1.".to_string(), "Document 2.".to_string()];

    let query_embeddings = model.encode(&query_sentences, true)?;
    let document_embeddings = model.encode(&document_sentences, false)?;

    let similarities = model.similarity(&query_embeddings, &document_embeddings)?;
    let score = similarities.data[0][0];

    println!("Similarity score: {}", score);

    // Hierarchical pooling reduces the dimensionality of document embeddings.
    // By setting the `pooling_factor` to 2, the number of token embeddings
    // is divided by 2. This can improve computational efficiency for downstream
    // tasks like similarity calculations by reducing the data size. You can
    // experiment with different `pooling_factor` values (e.g., 3, 4) based
    // on your specific requirements and the desired trade-off between
    // dimensionality reduction and information retention.
    let document_embeddings = hierarchical_pooling(&document_embeddings, 2)?;

    let similarities = model.similarity(&query_embeddings, &document_embeddings)?;
    let score = similarities.data[0][0];

    println!("Similarity score after hierarchical pooling: {}", score);

    Ok(())
}
