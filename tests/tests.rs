#![cfg(all(test, feature = "hf-hub"))]

use anyhow::Result;
use candle_core::Device;
use pylate_rs::{hierarchical_pooling, ColBERT};

/// Tests the `GTE-ModernColBERT-v1` model from the Hugging Face Hub.
#[test]
fn gte_modern_colbert_test() -> Result<()> {
    let device = Device::Cpu;
    println!("Testing with lightonai/GTE-ModernColBERT-v1...");

    let mut model: ColBERT = ColBERT::from("lightonai/GTE-ModernColBERT-v1")
        .with_device(device)
        .try_into()?;

    let query_sentences = vec!["what is the capital of france".to_string()];
    let document_sentences = vec!["paris is the capital of france".to_string()];

    let query_embeddings = model.encode(&query_sentences, true)?;
    let document_embeddings = model.encode(&document_sentences, false)?;

    let similarities = model.similarity(&query_embeddings, &document_embeddings)?;
    let score = similarities.data[0][0];

    println!("GTE-ModernColBERT-v1 Similarity: {}", score);
    let expected_score = 9.50805;
    let tolerance = 1e-2;
    assert!(
        (score - expected_score).abs() < tolerance,
        "Score {} is not within tolerance of {}",
        score,
        expected_score
    );

    let document_sentences = vec![
        "paris is the capital of france".to_string(),
        "berlin is the capital of germany, this is a test".to_string(),
    ];

    let document_embeddings = model.encode(&document_sentences, false)?;

    println!(
        "Documents embeddings shape: {:?}",
        document_embeddings.dims()
    );
    let document_embeddings = hierarchical_pooling(&document_embeddings, 2)?;
    println!(
        "Pooled documents embeddings shape: {:?}",
        document_embeddings.dims()
    );

    Ok(())
}

/// Tests the `colbertv2.0` model from the Hugging Face Hub.
#[test]
fn colbert_v2_test() -> Result<()> {
    let device = Device::Cpu; // Changed to CPU for broader compatibility
    println!("Testing with lightonai/colbertv2.0...");

    let mut model: ColBERT = ColBERT::from("lightonai/colbertv2.0")
        .with_device(device)
        .try_into()?;

    let query_sentences = vec!["what is the capital of france".to_string()];
    let document_sentences = vec!["paris is the capital of france".to_string()];

    let query_embeddings = model.encode(&query_sentences, true)?;
    let document_embeddings = model.encode(&document_sentences, false)?;

    let similarities = model.similarity(&query_embeddings, &document_embeddings)?;
    let score = similarities.data[0][0];

    println!("colbertv2.0 Similarity: {}", score);
    let expected_score = 29.603443;
    let tolerance = 1e-2;
    assert!(
        (score - expected_score).abs() < tolerance,
        "Score {} is not within tolerance of {}",
        score,
        expected_score
    );
    Ok(())
}

/// Tests the `answerai-colbert-small-v1` model from the Hugging Face Hub.
#[test]
fn answerai_colbert_small_v1_test() -> Result<()> {
    let device = Device::Cpu; // Changed to CPU for broader compatibility
    println!("Testing with lightonai/answerai-colbert-small-v1...");

    let mut model: ColBERT = ColBERT::from("lightonai/answerai-colbert-small-v1")
        .with_device(device)
        .try_into()?;

    let query_sentences = vec!["what is the capital of france".to_string()];
    let document_sentences = vec!["paris is the capital of france".to_string()];

    let query_embeddings = model.encode(&query_sentences, true)?;
    let document_embeddings = model.encode(&document_sentences, false)?;

    let similarities = model.similarity(&query_embeddings, &document_embeddings)?;
    let score = similarities.data[0][0];

    println!("answerai-colbert-small-v1 Similarity: {}", score);
    let expected_score = 31.490696;
    let tolerance = 1e-2;
    assert!(
        (score - expected_score).abs() < tolerance,
        "Score {} is not within tolerance of {}",
        score,
        expected_score
    );
    Ok(())
}
