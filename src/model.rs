use crate::{
    error::ColbertError,
    modernbert::{Config as ModernBertConfig, ModernBert},
    types::Similarities,
    utils::normalize_l2,
};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use tokenizers::Tokenizer;

/// Activation functions that can be applied after linear layers.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Activation {
    Identity,
    Relu,
    Gelu,
    GeluErf,
    Tanh,
    Silu,
}

impl Activation {
    /// Parse activation function from PyTorch class name
    fn from_pytorch_name(name: &str) -> Result<Self, ColbertError> {
        match name {
            "torch.nn.modules.linear.Identity" => Ok(Activation::Identity),
            "torch.nn.modules.activation.ReLU" => Ok(Activation::Relu),
            "torch.nn.modules.activation.GELU" => Ok(Activation::GeluErf),
            "torch.nn.modules.activation.Tanh" => Ok(Activation::Tanh),
            "torch.nn.modules.activation.SiLU" => Ok(Activation::Silu),
            _ => Err(ColbertError::Operation(format!(
                "Unsupported activation function: {}",
                name
            ))),
        }
    }

    /// Apply the activation function to a tensor
    fn apply(&self, tensor: &Tensor) -> Result<Tensor, candle_core::Error> {
        match self {
            Activation::Identity => Ok(tensor.clone()),
            Activation::Relu => tensor.relu(),
            Activation::Gelu => tensor.gelu(),
            Activation::GeluErf => tensor.gelu_erf(),
            Activation::Tanh => tensor.tanh(),
            Activation::Silu => tensor.silu(),
        }
    }
}

/// A single projection layer with its associated activation function.
#[derive(Clone)]
pub(crate) struct ProjectionLayer {
    linear: Linear,
    activation: Activation,
}

impl ProjectionLayer {
    /// Creates a new projection layer from config and weights
    fn new(
        config_bytes: &[u8],
        weights_bytes: Vec<u8>,
        layer_name: &str,
        device: &Device,
    ) -> Result<Self, ColbertError> {
        let config: serde_json::Value = serde_json::from_slice(config_bytes)?;
        let vb = VarBuilder::from_buffered_safetensors(weights_bytes, DType::F32, device)?;

        let in_features = config["in_features"]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| {
                ColbertError::Operation(format!("Missing 'in_features' in {} config", layer_name))
            })?;

        let out_features = config["out_features"]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| {
                ColbertError::Operation(format!("Missing 'out_features' in {} config", layer_name))
            })?;

        // Parse activation function from config, default to Identity if not specified
        let activation = config["activation_function"]
            .as_str()
            .map(Activation::from_pytorch_name)
            .transpose()?
            .unwrap_or(Activation::Identity);

        let linear = candle_nn::linear_no_bias(in_features, out_features, vb.pp("linear"))?;

        Ok(Self { linear, activation })
    }

    /// Apply the projection layer (linear + activation) to a tensor
    fn forward(&self, tensor: &Tensor) -> Result<Tensor, candle_core::Error> {
        let output = self.linear.forward(tensor)?;
        self.activation.apply(&output)
    }

    /// Get output dimension of this layer
    fn out_features(&self) -> usize {
        self.linear.weight().dims()[0]
    }

    /// Get input dimension of this layer
    fn in_features(&self) -> usize {
        self.linear.weight().dims()[1]
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use rayon::prelude::*;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use crate::builder::ColbertBuilder;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// An enum to abstract over different underlying BERT-based models.
///
/// This allows `ColBERT` to use different architectures like
/// `BertModel` or `ModernBert` without changing the core logic.
pub enum BaseModel {
    /// A variant holding a `ModernBert` model.
    ModernBert(ModernBert),
    /// A variant holding a standard `BertModel`.
    Bert(BertModel),
}

impl BaseModel {
    /// Performs a forward pass through the appropriate underlying model.
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        match self {
            BaseModel::ModernBert(model) => model.forward(input_ids, attention_mask),
            BaseModel::Bert(model) => {
                model.forward(input_ids, token_type_ids, Some(attention_mask))
            },
        }
    }
}

/// The main ColBERT model structure.
///
/// This struct encapsulates the language model, a linear projection layer,
/// the tokenizer, and all necessary configuration for performing encoding
/// and similarity calculations based on the ColBERT architecture.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct ColBERT {
    pub(crate) model: BaseModel,
    pub(crate) projection_layers: Vec<ProjectionLayer>,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) mask_token_id: u32,
    pub(crate) mask_token: String,
    pub(crate) query_prefix: String,
    pub(crate) document_prefix: String,
    pub(crate) do_query_expansion: bool,
    pub(crate) attend_to_expansion_tokens: bool,
    pub(crate) query_length: usize,
    pub(crate) document_length: usize,
    pub(crate) batch_size: usize,
    /// The device (CPU or GPU) on which the model is loaded.
    #[cfg_attr(feature = "wasm", wasm_bindgen(skip))]
    pub device: Device,
}

impl ColBERT {
    /// Creates a new instance of the `ColBERT` model from byte buffers.
    ///
    /// # Arguments
    /// * `projection_layers_data` - Vec of (config_bytes, weights_bytes, layer_name) tuples for each Dense layer
    pub fn new(
        weights: Vec<u8>,
        projection_layers_data: Vec<(Vec<u8>, Vec<u8>, String)>,
        tokenizer_bytes: Vec<u8>,
        config_bytes: Vec<u8>,
        query_prefix: String,
        document_prefix: String,
        mask_token: String,
        do_query_expansion: bool,
        attend_to_expansion_tokens: bool,
        query_length: Option<usize>,
        document_length: Option<usize>,
        batch_size: Option<usize>,
        device: &Device,
    ) -> Result<Self, ColbertError> {
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, device)?;

        let config_value: serde_json::Value = serde_json::from_slice(&config_bytes)?;
        let architectures = config_value["architectures"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ColbertError::Operation("Missing or invalid 'architectures' in config.json".into())
            })?;

        let model = match architectures {
            "ModernBertModel" => {
                let config: ModernBertConfig = serde_json::from_slice(&config_bytes)?;
                let model = ModernBert::load(vb.clone(), &config)?;
                BaseModel::ModernBert(model)
            },
            "BertForMaskedLM" | "BertModel" => {
                let config: BertConfig = serde_json::from_slice(&config_bytes)?;
                let model = BertModel::load(vb.clone(), &config)?;
                BaseModel::Bert(model)
            },
            arch => {
                return Err(ColbertError::Operation(format!(
                    "Unsupported architecture: {}",
                    arch
                )))
            },
        };

        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)?;

        let mask_token_id = tokenizer.token_to_id(mask_token.as_str()).ok_or_else(|| {
            ColbertError::Operation(format!(
                "Token '{}' not found in the tokenizer's vocabulary.",
                mask_token
            ))
        })?;

        // Validate that we have at least one projection layer
        if projection_layers_data.is_empty() {
            return Err(ColbertError::Operation(
                "At least one projection layer is required".into(),
            ));
        }

        // Build projection layers and validate dimensions
        let mut projection_layers = Vec::with_capacity(projection_layers_data.len());
        let mut prev_out_features: Option<usize> = None;

        for (i, (config_bytes, weights_bytes, layer_name)) in
            projection_layers_data.into_iter().enumerate()
        {
            let layer = ProjectionLayer::new(&config_bytes, weights_bytes, &layer_name, device)?;

            // Validate dimension compatibility with previous layer
            if let Some(prev_out) = prev_out_features {
                if prev_out != layer.in_features() {
                    return Err(ColbertError::Operation(format!(
                        "Dimension mismatch between layer {} and {}: output {} != input {}",
                        i - 1,
                        layer_name,
                        prev_out,
                        layer.in_features()
                    )));
                }
            }

            prev_out_features = Some(layer.out_features());
            projection_layers.push(layer);
        }

        // If do_query_expansion is false, attend_to_expansion_tokens should also be false
        let final_attend_to_expansion_tokens = if !do_query_expansion {
            false
        } else {
            attend_to_expansion_tokens
        };

        Ok(Self {
            model,
            projection_layers,
            tokenizer,
            mask_token_id,
            mask_token,
            query_prefix,
            document_prefix,
            do_query_expansion,
            attend_to_expansion_tokens: final_attend_to_expansion_tokens,
            query_length: query_length.unwrap_or(32),
            document_length: document_length.unwrap_or(180),
            batch_size: batch_size.unwrap_or(32),
            device: device.clone(),
        })
    }

    /// Creates a `ColbertBuilder` to construct a `ColBERT` model from a Hugging Face repository.
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    pub fn from(repo_id: &str) -> ColbertBuilder {
        ColbertBuilder::new(repo_id)
    }

    /// Processes document embeddings by filtering based on an attention mask,
    /// normalizing the results, and padding them to a uniform length.
    ///
    /// This method iterates through each embedding in the batch,
    /// removes vectors corresponding to padding tokens (where attention_mask is 0),
    /// normalizes the remaining vectors, and then pads all sequences with zeros to match
    /// the longest sequence in the batch.
    fn filter_normalize_and_pad(
        &self,
        embeddings: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        let (batch_size, _, dim) = embeddings.dims3()?;
        let mut processed_embeddings: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut max_len = 0;

        // Iterate over each item in the batch.
        for i in 0..batch_size {
            let single_embedding = embeddings.i(i)?;
            let single_mask = attention_mask.i(i)?.to_vec1::<u32>()?;

            // Collect embedding vectors where the attention mask is 1.
            let mut kept_rows = Vec::new();
            for (j, &mask_val) in single_mask.iter().enumerate() {
                if mask_val == 1 {
                    kept_rows.push(single_embedding.i(j)?);
                }
            }

            // Normalize the filtered embeddings.
            let (normalized, current_len) = if kept_rows.is_empty() {
                // If all tokens are masked, produce a single zero vector. This avoids errors
                // with empty tensors and provides a valid, though empty, representation.
                let zeros = Tensor::zeros((1, dim), DType::F32, &self.device)?;
                (zeros, 1)
            } else {
                let filtered = Tensor::stack(&kept_rows, 0)?;
                let len = filtered.dim(0)?;
                (normalize_l2(&filtered)?, len)
            };

            // Track the maximum sequence length for padding later.
            if current_len > max_len {
                max_len = current_len;
            }
            processed_embeddings.push(normalized);
        }

        // Pad all processed embeddings to the maximum length found in the batch.
        let mut padded_tensors = Vec::with_capacity(batch_size);
        for tensor in processed_embeddings.iter() {
            let current_len = tensor.dim(0)?;
            let dim = tensor.dim(1)?; // Should be the same for all
            let pad_len = max_len - current_len;

            if pad_len > 0 {
                let padding = Tensor::zeros((pad_len, dim), DType::F32, &self.device)?;
                let padded = Tensor::cat(&[tensor, &padding], 0)?;
                padded_tensors.push(padded);
            } else {
                padded_tensors.push(tensor.clone());
            }
        }

        // Stack the padded tensors into a single batch tensor.
        Tensor::stack(&padded_tensors, 0)
    }

    /// Encodes a batch of sentences (queries or documents) into embeddings.
    ///
    /// On CPU and non-WASM targets, this method leverages Rayon for parallel batch processing
    /// to accelerate encoding. On other targets (like GPU or WASM), it processes
    /// batches sequentially.
    pub fn encode(&mut self, sentences: &[String], is_query: bool) -> Result<Tensor, ColbertError> {
        if sentences.is_empty() {
            return Err(ColbertError::Operation(
                "Input sentences cannot be empty.".into(),
            ));
        }

        // Use Rayon for parallel processing on CPU, but not on WASM.
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        if self.device.is_cpu() {
            let mut tokenized_batches = Vec::new();
            for batch_sentences in sentences.chunks(self.batch_size) {
                tokenized_batches.push(self.tokenize(batch_sentences, is_query)?);
            }

            let all_embeddings = tokenized_batches
                .into_par_iter()
                .map(
                    |(token_ids, attention_mask, token_type_ids)| -> Result<Tensor, ColbertError> {
                        let token_embeddings =
                            self.model
                                .forward(&token_ids, &attention_mask, &token_type_ids)?;

                        // Apply all projection layers sequentially
                        let mut projected_embeddings = token_embeddings;
                        for layer in &self.projection_layers {
                            projected_embeddings = layer.forward(&projected_embeddings)?;
                        }

                        if !self.do_query_expansion || !is_query {
                            // Apply filtering, normalization, and padding.
                            self.filter_normalize_and_pad(&projected_embeddings, &attention_mask)
                                .map_err(ColbertError::from)
                        } else {
                            // Original behavior: just normalize.
                            normalize_l2(&projected_embeddings)
                        }
                    },
                )
                .collect::<Result<Vec<_>, _>>()?;

            if all_embeddings.len() == 1 {
                return Ok(all_embeddings.into_iter().next().unwrap());
            }
            return Tensor::cat(&all_embeddings, 0).map_err(ColbertError::from);
        }

        // Fallback to sequential processing for GPU, WASM, or other devices.
        let mut all_embeddings = Vec::new();
        for batch_sentences in sentences.chunks(self.batch_size) {
            let (token_ids, attention_mask, token_type_ids) =
                self.tokenize(batch_sentences, is_query)?;

            let token_embeddings =
                self.model
                    .forward(&token_ids, &attention_mask, &token_type_ids)?;

            // Apply all projection layers sequentially
            let mut projected_embeddings = token_embeddings;
            for layer in &self.projection_layers {
                projected_embeddings = layer.forward(&projected_embeddings)?;
            }

            let final_embeddings = if !self.do_query_expansion || !is_query {
                // Apply filtering, normalization, and padding.
                self.filter_normalize_and_pad(&projected_embeddings, &attention_mask)?
            } else {
                // Original behavior: just normalize.
                normalize_l2(&projected_embeddings)?
            };

            all_embeddings.push(final_embeddings);
        }

        if all_embeddings.len() == 1 {
            return Ok(all_embeddings.remove(0));
        }

        Tensor::cat(&all_embeddings, 0).map_err(ColbertError::from)
    }

    /// Calculates the similarity scores between query and document embeddings.
    pub fn similarity(
        &self,
        queries_embeddings: &Tensor,
        documents_embeddings: &Tensor,
    ) -> Result<Similarities, ColbertError> {
        let scores = queries_embeddings
            .unsqueeze(1)?
            .broadcast_matmul(&documents_embeddings.transpose(1, 2)?.unsqueeze(0)?)?;

        let max_scores = scores.max(3)?;
        let similarities = max_scores.sum(2)?;
        let similarities_vec = similarities.to_vec2::<f32>()?;
        Ok(Similarities {
            data: similarities_vec,
        })
    }

    /// Computes the raw, un-reduced similarity matrix between query and document embeddings.
    pub fn raw_similarity(
        &self,
        queries_embeddings: &Tensor,
        documents_embeddings: &Tensor,
    ) -> Result<Tensor, ColbertError> {
        queries_embeddings
            .unsqueeze(1)?
            .broadcast_matmul(&documents_embeddings.transpose(1, 2)?.unsqueeze(0)?)
            .map_err(ColbertError::from)
    }

    /// Tokenizes a batch of texts, applying specific logic for queries and documents.
    pub(crate) fn tokenize(
        &mut self,
        texts: &[String],
        is_query: bool,
    ) -> Result<(Tensor, Tensor, Tensor), ColbertError> {
        let device = &self.device;
        let (prefix, max_length) = if is_query {
            (self.query_prefix.as_str(), self.query_length)
        } else {
            (self.document_prefix.as_str(), self.document_length)
        };

        // Prepend the appropriate prefix to each text to create the full input strings.
        // This simplifies the logic by allowing the tokenizer to handle the entire sequence at once,
        // avoiding manual and complex tensor concatenation.
        let texts_with_prefix: Vec<_> = texts
            .iter()
            .map(|text| format!("{}{}", prefix, text))
            .collect();

        // Configure tokenizer truncation to the model's maximum sequence length.
        let _ = self
            .tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length,
                ..Default::default()
            }));

        // Configure the padding strategy based on the input type.
        let padding_params = if is_query {
            // For ColBERT queries, pad to a fixed length with the [MASK] token.
            tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(max_length),
                pad_id: self.mask_token_id,
                pad_token: self.mask_token.clone(),
                ..Default::default()
            }
        } else {
            // Documents are padded to the longest sequence in the batch.
            tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            }
        };
        self.tokenizer.with_padding(Some(padding_params));

        // Tokenize the batch of prepared texts.
        let encodings = self.tokenizer.encode_batch(texts_with_prefix, true)?;

        let batch_size = encodings.len();
        if batch_size == 0 {
            return Err(ColbertError::Operation(
                "Input sentences cannot be empty.".into(),
            ));
        }

        // Collect tokenization outputs into flat vectors.
        let seq_len = encodings.first().map_or(0, |e| e.get_ids().len());
        let (mut ids_vec, mut mask_vec, mut type_ids_vec) =
            (Vec::<u32>::new(), Vec::<u32>::new(), Vec::<u32>::new());
        for enc in &encodings {
            ids_vec.extend(enc.get_ids());
            mask_vec.extend(enc.get_attention_mask());
            type_ids_vec.extend(enc.get_type_ids());
        }

        // Create tensors from the vectors.
        let token_ids = Tensor::from_vec(ids_vec, (batch_size, seq_len), device)?;
        let mut attention_mask = Tensor::from_vec(mask_vec, (batch_size, seq_len), device)?;
        let token_type_ids = Tensor::from_vec(type_ids_vec, (batch_size, seq_len), device)?;

        // For queries, optionally set the attention mask to all ones to attend to padded tokens.
        if is_query && self.attend_to_expansion_tokens {
            attention_mask = attention_mask.ones_like()?;
        }

        Ok((token_ids, attention_mask, token_type_ids))
    }
}
