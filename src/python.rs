use crate::{error::ColbertError, model::ColBERT};
use candle_core::{Device, Tensor};
use ndarray::Array;
use numpy::{ndarray::IxDyn, PyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::{exceptions::PyValueError, types::PyModule, Bound};
use std::convert::TryFrom;

use crate::pooling::hierarchical_pooling;

// Custom Python exception for Colbert errors
pyo3::create_exception!(pylate_rs, ColbertException, pyo3::exceptions::PyException);

impl From<ColbertError> for PyErr {
    fn from(err: ColbertError) -> PyErr {
        ColbertException::new_err(err.to_string())
    }
}

/// A Python wrapper for the Rust ColBERT model.
///
/// This class provides a Python interface to the underlying Rust implementation,
/// allowing for model loading, encoding, and similarity calculations.
#[pyclass(name = "PyColBERT")]
pub struct PyColBERT {
    model: ColBERT,
}

#[pymethods]
impl PyColBERT {
    /// Creates a new `PyColBERT` instance by loading a pretrained model
    /// from the Hugging Face Hub.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - The identifier of the model repository on the Hugging Face Hub.
    /// * `device` - The device to run the model on ("cpu", "cuda", "cuda:integer", or "mps").
    /// * `query_length` - The maximum length for queries.
    /// * `document_length` - The maximum length for documents.
    /// * `batch_size` - The batch size for encoding.
    /// * `do_query_expansion` - Whether to perform query expansion.
    /// * `attend_to_expansion_tokens` - Whether to attend to expansion tokens.
    /// * `query_prefix` - The prefix to add to queries.
    /// * `document_prefix` - The prefix to add to documents.
    /// * `mask_token` - The mask token to use for padding queries.
    ///
    /// # Returns
    ///
    /// A new instance of `PyColBERT`.
    #[staticmethod]
    #[pyo3(signature = (
        repo_id,
        device=None,
        query_length=None,
        document_length=None,
        batch_size=None,
        do_query_expansion=None,
        attend_to_expansion_tokens=None,
        query_prefix=None,
        document_prefix=None,
        mask_token=None
    ))]
    pub fn from_pretrained(
        repo_id: &str,
        device: Option<&str>,
        query_length: Option<usize>,
        document_length: Option<usize>,
        batch_size: Option<usize>,
        do_query_expansion: Option<bool>,
        attend_to_expansion_tokens: Option<bool>,
        query_prefix: Option<String>,
        document_prefix: Option<String>,
        mask_token: Option<String>,
    ) -> PyResult<Self> {
        let device = match device {
            Some(device_str) if device_str.starts_with("cuda") => {
                let parts: Vec<&str> = device_str.split(':').collect();
                let device_index = if parts.len() == 2 {
                    parts[1]
                        .parse::<usize>()
                        .map_err(|_| PyValueError::new_err("Invalid CUDA device index"))?
                } else {
                    0
                };
                Device::new_cuda(device_index).map_err(|e| PyValueError::new_err(e.to_string()))?
            },
            Some("mps") => {
                Device::new_metal(0).map_err(|e| PyValueError::new_err(e.to_string()))?
            },
            _ => Device::Cpu,
        };

        let mut builder = ColBERT::from(repo_id).with_device(device);

        if let Some(ql) = query_length {
            builder = builder.with_query_length(ql);
        }
        if let Some(dl) = document_length {
            builder = builder.with_document_length(dl);
        }
        if let Some(bs) = batch_size {
            builder = builder.with_batch_size(bs);
        }
        if let Some(do_expansion) = do_query_expansion {
            builder = builder.with_do_query_expansion(do_expansion);
        }
        if let Some(attend) = attend_to_expansion_tokens {
            builder = builder.with_attend_to_expansion_tokens(attend);
        }
        if let Some(qp) = query_prefix {
            builder = builder.with_query_prefix(qp);
        }
        if let Some(dp) = document_prefix {
            builder = builder.with_document_prefix(dp);
        }
        if let Some(mt) = mask_token {
            builder = builder.with_mask_token(mt);
        }

        let model = ColBERT::try_from(builder)?;
        Ok(Self { model })
    }

    /// Encodes a list of sentences (queries or documents) into embeddings.
    ///
    /// # Arguments
    ///
    /// * `sentences` - A list of strings to encode.
    /// * `is_query` - A boolean flag indicating whether the sentences are queries (`true`) or documents (`false`).
    ///
    /// # Returns
    ///
    /// A NumPy array containing the embeddings.
    pub fn encode<'py>(
        &mut self,
        py: Python<'py>,
        sentences: Vec<String>,
        is_query: bool,
        pool_factor: usize,
    ) -> PyResult<Bound<'py, PyArray<f32, IxDyn>>> {
        let embeddings = self.model.encode(&sentences, is_query)?;

        let embeddings = if pool_factor > 1 {
            hierarchical_pooling(&embeddings, pool_factor)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            embeddings
        };

        let shape = embeddings.dims();
        let data = embeddings
            .flatten_all()
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let ndarray = Array::from_shape_vec(shape, data)
            .map_err(|e| PyValueError::new_err(format!("Error creating ndarray: {}", e)))?;

        Ok(PyArray::from_owned_array(py, ndarray))
    }

    /// Calculates similarity scores between query and document embeddings.
    ///
    /// # Arguments
    ///
    /// * `queries_embeddings` - A NumPy array of query embeddings.
    /// * `documents_embeddings` - A NumPy array of document embeddings.
    ///
    /// # Returns
    ///
    /// A nested list of f32 similarity scores.
    pub fn similarity(
        &self,
        queries_embeddings: PyReadonlyArrayDyn<f32>,
        documents_embeddings: PyReadonlyArrayDyn<f32>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let queries_tensor = tensor_from_array(queries_embeddings, &self.model.device)?;
        let documents_tensor = tensor_from_array(documents_embeddings, &self.model.device)?;

        let similarities = self.model.similarity(&queries_tensor, &documents_tensor)?;
        Ok(similarities.data)
    }
}

/// Helper function to convert a NumPy array to a Candle tensor.
fn tensor_from_array(array: PyReadonlyArrayDyn<f32>, device: &Device) -> PyResult<Tensor> {
    let shape = array.shape();
    let data = array.as_slice()?;
    Tensor::from_vec(data.to_vec(), shape, device).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The main Python module definition.
///
/// This module, named `pylate_rs`, exposes the `PyColBERT` class to Python.
#[pymodule]
fn pylate_rs<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<PyColBERT>()?;
    Ok(())
}
