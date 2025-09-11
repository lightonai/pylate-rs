use crate::{
    error::ColbertError,
    model::ColBERT,
    pooling::hierarchical_pooling,
    types::{EncodeInput, EncodeOutput, RawSimilarityOutput, SimilarityInput},
};
use candle_core::{Device, IndexOp, Tensor};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl ColBERT {
    /// WASM-compatible constructor.
    #[wasm_bindgen(constructor)]
    pub fn from_bytes(
        weights: Vec<u8>,
        dense_weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
        sentence_transformers_config: Vec<u8>,
        dense_config: Vec<u8>,
        special_tokens_map: Vec<u8>,
        batch_size: Option<usize>,
    ) -> Result<ColBERT, JsValue> {
        console_error_panic_hook::set_once();

        let st_config: serde_json::Value =
            serde_json::from_slice(&sentence_transformers_config).map_err(ColbertError::from)?;

        let special_tokens_map: serde_json::Value =
            serde_json::from_slice(&special_tokens_map).map_err(ColbertError::from)?;

        let query_prefix = st_config["query_prefix"]
            .as_str()
            .unwrap_or("[Q]")
            .to_string();
        let document_prefix = st_config["document_prefix"]
            .as_str()
            .unwrap_or("[D]")
            .to_string();
        let do_query_expansion = st_config["do_query_expansion"]
            .as_bool()
            .unwrap_or(true);
        let attend_to_expansion_tokens = st_config["attend_to_expansion_tokens"]
            .as_bool()
            .unwrap_or(false);
        let query_length = st_config["query_length"].as_u64().map(|v| v as usize);
        let document_length = st_config["document_length"].as_u64().map(|v| v as usize);

        let mask_token = special_tokens_map["mask_token"]
            .as_str()
            .unwrap_or("[MASK]")
            .to_string();

        let batch_size = Some(batch_size.unwrap_or(32));

        Self::new(
            weights,
            dense_weights,
            tokenizer,
            config,
            dense_config,
            query_prefix,
            document_prefix,
            mask_token,
            do_query_expansion,
            attend_to_expansion_tokens,
            query_length,
            document_length,
            batch_size,
            &Device::Cpu,
        )
        .map_err(Into::into)
    }

    /// WASM-compatible version of the `encode` method.
    #[wasm_bindgen(js_name = "encode")]
    pub fn encode_wasm(&mut self, input: JsValue, is_query: bool) -> Result<JsValue, JsValue> {
        let params: EncodeInput = serde_wasm_bindgen::from_value(input)?;
        // Override model's batch_size if provided in the input
        if let Some(batch_size) = params.batch_size {
            self.batch_size = batch_size;
        }
        let embeddings_tensor = self.encode(&params.sentences, is_query)?;
        let embeddings_data = embeddings_tensor
            .to_vec3::<f32>()
            .map_err(ColbertError::from)?;
        let result = EncodeOutput {
            embeddings: embeddings_data,
        };
        serde_wasm_bindgen::to_value(&result).map_err(Into::into)
    }

    /// WASM-compatible version of the `similarity` method.
    #[wasm_bindgen(js_name = "similarity")]
    pub fn similarity_wasm(&mut self, input: JsValue) -> Result<JsValue, JsValue> {
        let params: SimilarityInput = serde_wasm_bindgen::from_value(input)?;
        let queries_embeddings = self.encode(&params.queries, true)?;
        let documents_embeddings = self.encode(&params.documents, false)?;
        let result = self.similarity(&queries_embeddings, &documents_embeddings)?;
        serde_wasm_bindgen::to_value(&result).map_err(Into::into)
    }

    /// WASM-compatible method to get the raw similarity matrix and tokens.
    #[wasm_bindgen(js_name = "raw_similarity_matrix")]
    pub fn raw_similarity_matrix_wasm(&mut self, input: JsValue) -> Result<JsValue, JsValue> {
        let params: SimilarityInput = serde_wasm_bindgen::from_value(input)?;

        let (query_ids_tensor, _, _) = self.tokenize(&params.queries, true)?;
        let query_ids_vec: Vec<Vec<u32>> =
            query_ids_tensor.to_vec2().map_err(ColbertError::from)?;
        let query_tokens: Vec<Vec<String>> = query_ids_vec
            .iter()
            .map(|ids| {
                ids.iter()
                    .map(|&id| self.tokenizer.id_to_token(id).unwrap_or_default())
                    .collect()
            })
            .collect();

        let (doc_ids_tensor, _, _) = self.tokenize(&params.documents, false)?;
        let doc_ids_vec: Vec<Vec<u32>> = doc_ids_tensor.to_vec2().map_err(ColbertError::from)?;
        let document_tokens: Vec<Vec<String>> = doc_ids_vec
            .iter()
            .map(|ids| {
                ids.iter()
                    .map(|&id| self.tokenizer.id_to_token(id).unwrap_or_default())
                    .collect()
            })
            .collect();

        let queries_embeddings = self.encode(&params.queries, true)?;
        let documents_embeddings = self.encode(&params.documents, false)?;

        let scores_tensor = self.raw_similarity(&queries_embeddings, &documents_embeddings)?;

        let (dim_q, dim_d, _, _) = scores_tensor.dims4().map_err(ColbertError::from)?;
        let mut scores_vec: Vec<Vec<Vec<Vec<f32>>>> = Vec::with_capacity(dim_q);
        for i in 0..dim_q {
            let mut docs_vec: Vec<Vec<Vec<f32>>> = Vec::with_capacity(dim_d);
            for j in 0..dim_d {
                let matrix_2d = scores_tensor.i((i, j)).map_err(ColbertError::from)?;
                let matrix_vec = matrix_2d.to_vec2::<f32>().map_err(ColbertError::from)?;
                docs_vec.push(matrix_vec);
            }
            scores_vec.push(docs_vec);
        }

        let result = RawSimilarityOutput {
            similarity_matrix: scores_vec,
            query_tokens,
            document_tokens,
        };

        serde_wasm_bindgen::to_value(&result).map_err(Into::into)
    }
}

#[cfg(feature = "wasm")]
#[derive(serde::Deserialize)]
struct PoolingInput {
    embeddings: Vec<Vec<Vec<f32>>>,
    pool_factor: usize,
}

/// WASM-compatible version of the `hierarchical_pooling` function.
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = hierarchical_pooling)]
pub fn hierarchical_pooling_wasm(input: JsValue) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let params: PoolingInput = serde_wasm_bindgen::from_value(input)?;

    if params.embeddings.is_empty() {
        let result = EncodeOutput { embeddings: vec![] };
        return serde_wasm_bindgen::to_value(&result).map_err(Into::into);
    }

    let batch_size = params.embeddings.len();
    let n_tokens = params.embeddings[0].len();

    if n_tokens == 0 {
        let result = EncodeOutput {
            embeddings: params.embeddings,
        };
        return serde_wasm_bindgen::to_value(&result).map_err(Into::into);
    }
    let embedding_dim = params.embeddings[0][0].len();

    let flat_embeddings: Vec<f32> = params.embeddings.into_iter().flatten().flatten().collect();
    let documents_embeddings = Tensor::from_vec(
        flat_embeddings,
        (batch_size, n_tokens, embedding_dim),
        &Device::Cpu,
    )
    .map_err(ColbertError::from)?;

    // Call the original Rust function, mapping the anyhow::Error to a ColbertError.
    let pooled_tensor = hierarchical_pooling(&documents_embeddings, params.pool_factor)
        .map_err(|e| ColbertError::Operation(e.to_string()))?;

    let embeddings_data = pooled_tensor.to_vec3::<f32>().map_err(ColbertError::from)?;
    let result = EncodeOutput {
        embeddings: embeddings_data,
    };

    serde_wasm_bindgen::to_value(&result).map_err(Into::into)
}
