use candle_core::{DType, Tensor};
use candle_onnx::onnx::ModelProto;
use candle_onnx::simple_eval;
use std::collections::HashMap;

use crate::error::ColbertError;

/// A wrapper around an ONNX model that resolves input/output names
/// from the graph metadata at construction time.
pub struct OnnxModel {
    /// The raw ONNX model proto.
    model: ModelProto,
    /// Map from semantic role ("input_ids", "attention_mask") to the actual
    /// input name in the ONNX graph.
    input_names: HashMap<String, String>,
    /// Ordered list of output names from the ONNX graph.
    output_names: Vec<String>,
}

impl OnnxModel {
    /// Creates a new `OnnxModel` by reading input/output names from the ONNX graph.
    ///
    /// Input names are matched by looking for known patterns in the graph's
    /// input definitions (e.g., names containing "input_id" or "attention_mask").
    /// This avoids hardcoding names that may vary across different ONNX exports.
    pub fn new(model: ModelProto) -> Result<Self, ColbertError> {
        // Get the graph, or error if it's missing
        let graph = model.graph.as_ref().ok_or_else(|| {
            ColbertError::Operation("ONNX model has no graph defined".into())
        })?;

        // Discover input names by matching known patterns
        let mut input_names = HashMap::new();
        for input in &graph.input {
            let name = &input.name;
            let lower = name.to_lowercase();
            if lower.contains("input_id") {
                input_names.insert("input_ids".to_string(), name.clone());
            } else if lower.contains("attention_mask") {
                input_names.insert("attention_mask".to_string(), name.clone());
            } else if lower.contains("token_type") {
                input_names.insert("token_type_ids".to_string(), name.clone());
            }
        }

        // Verify that we found the required inputs
        if !input_names.contains_key("input_ids") {
            return Err(ColbertError::Operation(
                format!(
                    "Could not find 'input_ids' in ONNX graph inputs: {:?}",
                    graph.input.iter().map(|i| &i.name).collect::<Vec<_>>()
                )
            ));
        }

        // Collect all output names
        let output_names: Vec<String> = graph
            .output
            .iter()
            .map(|o| o.name.clone())
            .collect();

        if output_names.is_empty() {
            return Err(ColbertError::Operation(
                "ONNX model has no outputs defined in graph".into(),
            ));
        }

        Ok(Self {
            model,
            input_names,
            output_names,
        })
    }

    /// Performs a forward pass through the ONNX model.
    ///
    /// Converts inputs to I64 (required by most ONNX models), maps them
    /// to the correct graph input names, and returns the last output tensor.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        let mut params = HashMap::new();

        // Map our tensors to the actual ONNX graph input names
        let ids_name = self.input_names.get("input_ids")
            .ok_or_else(|| candle_core::Error::Msg("Missing input_ids mapping".into()))?;
        params.insert(ids_name.clone(), input_ids.to_dtype(DType::I64)?);

        if let Some(mask_name) = self.input_names.get("attention_mask") {
            params.insert(mask_name.clone(), attention_mask.to_dtype(DType::I64)?);
        }

        // Run inference
        let result = simple_eval(&self.model, params)?;

        // Return the last output (typically the final hidden states)
        let output_name = self.output_names.last()
            .ok_or_else(|| candle_core::Error::Msg("No output names available".into()))?;

        result
            .get(output_name)
            .cloned()
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "Output '{}' not found in ONNX eval result",
                    output_name
                ))
            })
    }
}
