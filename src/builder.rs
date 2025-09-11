use crate::{error::ColbertError, model::ColBERT};
use candle_core::Device;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::{convert::TryFrom, fs, path::PathBuf};

/// A builder for configuring and creating a `ColBERT` model from the Hugging Face Hub.
///
/// This struct provides an interface to set various configuration options
/// before downloading the model files and initializing the `ColBERT` instance.
/// This is only available when the `hf-hub` feature is enabled.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub struct ColbertBuilder {
    repo_id: String,
    query_prefix: Option<String>,
    document_prefix: Option<String>,
    mask_token: Option<String>,
    do_query_expansion: Option<bool>,
    attend_to_expansion_tokens: Option<bool>,
    query_length: Option<usize>,
    document_length: Option<usize>,
    batch_size: Option<usize>,
    device: Option<Device>,
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl ColbertBuilder {
    /// Creates a new `ColbertBuilder`.
    pub(crate) fn new(repo_id: &str) -> Self {
        Self {
            repo_id: repo_id.to_string(),
            query_prefix: None,
            document_prefix: None,
            mask_token: None,
            do_query_expansion: None,
            attend_to_expansion_tokens: None,
            query_length: None,
            document_length: None,
            batch_size: None,
            device: None,
        }
    }

    /// Sets the query prefix token. Overrides the value from the config file.
    pub fn with_query_prefix(mut self, query_prefix: String) -> Self {
        self.query_prefix = Some(query_prefix);
        self
    }

    /// Sets the document prefix token. Overrides the value from the config file.
    pub fn with_document_prefix(mut self, document_prefix: String) -> Self {
        self.document_prefix = Some(document_prefix);
        self
    }

    /// Sets the mask token. Overrides the value from the `special_tokens_map.json` file.
    pub fn with_mask_token(mut self, mask_token: String) -> Self {
        self.mask_token = Some(mask_token);
        self
    }

    /// Sets whether to perform query expansion. Overrides the value from the config file.
    pub fn with_do_query_expansion(mut self, do_expansion: bool) -> Self {
        self.do_query_expansion = Some(do_expansion);
        self
    }

    /// Sets whether to attend to expansion tokens. Overrides the value from the config file.
    pub fn with_attend_to_expansion_tokens(mut self, attend: bool) -> Self {
        self.attend_to_expansion_tokens = Some(attend);
        self
    }

    /// Sets the maximum query length. Overrides the value from the config file.
    pub fn with_query_length(mut self, query_length: usize) -> Self {
        self.query_length = Some(query_length);
        self
    }

    /// Sets the maximum document length. Overrides the value from the config file.
    pub fn with_document_length(mut self, document_length: usize) -> Self {
        self.document_length = Some(document_length);
        self
    }

    /// Sets the batch size for encoding. Defaults to 32.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Sets the device to run the model on.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl TryFrom<ColbertBuilder> for ColBERT {
    type Error = ColbertError;

    /// Builds the `ColBERT` model by downloading files from the hub and initializing the model.
    fn try_from(builder: ColbertBuilder) -> Result<Self, Self::Error> {
        let device = builder.device.unwrap_or(Device::Cpu);

        let local_path = PathBuf::from(&builder.repo_id);
        let (
            tokenizer_path,
            weights_path,
            config_path,
            st_config_path,
            dense_config_path,
            dense_weights_path,
            special_tokens_map_path,
        ) = if local_path.is_dir() {
            (
                local_path.join("tokenizer.json"),
                local_path.join("model.safetensors"),
                local_path.join("config.json"),
                local_path.join("config_sentence_transformers.json"),
                local_path.join("1_Dense/config.json"),
                local_path.join("1_Dense/model.safetensors"),
                local_path.join("special_tokens_map.json"),
            )
        } else {
            let api = Api::new()?;
            let repo = api.repo(Repo::with_revision(
                builder.repo_id.clone(),
                RepoType::Model,
                "main".to_string(),
            ));
            (
                repo.get("tokenizer.json")?,
                repo.get("model.safetensors")?,
                repo.get("config.json")?,
                repo.get("config_sentence_transformers.json")?,
                repo.get("1_Dense/config.json")?,
                repo.get("1_Dense/model.safetensors")?,
                repo.get("special_tokens_map.json")?,
            )
        };

        if local_path.is_dir() {
            for path in [
                &tokenizer_path,
                &weights_path,
                &config_path,
                &st_config_path,
                &dense_config_path,
                &dense_weights_path,
                &special_tokens_map_path,
            ] {
                if !path.exists() {
                    return Err(ColbertError::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("File not found in local directory: {}", path.display()),
                    )));
                }
            }
        }

        let tokenizer_bytes = fs::read(tokenizer_path)?;
        let weights_bytes = fs::read(weights_path)?;
        let config_bytes = fs::read(config_path)?;
        let st_config_bytes = fs::read(st_config_path)?;
        let dense_config_bytes = fs::read(dense_config_path)?;
        let dense_weights_bytes = fs::read(dense_weights_path)?;
        let special_tokens_map_bytes = fs::read(special_tokens_map_path)?;

        let st_config: serde_json::Value = serde_json::from_slice(&st_config_bytes)?;
        let special_tokens_map: serde_json::Value =
            serde_json::from_slice(&special_tokens_map_bytes)?;

        let final_query_prefix = builder.query_prefix.unwrap_or_else(|| {
            st_config["query_prefix"]
                .as_str()
                .unwrap_or("[Q]")
                .to_string()
        });
        let final_document_prefix = builder.document_prefix.unwrap_or_else(|| {
            st_config["document_prefix"]
                .as_str()
                .unwrap_or("[D]")
                .to_string()
        });

        let mask_token = builder.mask_token.unwrap_or_else(|| {
            special_tokens_map["mask_token"]
                .as_str()
                .unwrap_or("[MASK]")
                .to_string()
        });

        let final_do_query_expansion = builder.do_query_expansion.unwrap_or_else(|| {
            st_config["do_query_expansion"]
                .as_bool()
                .unwrap_or(true)
        });

        let final_attend_to_expansion_tokens =
            builder.attend_to_expansion_tokens.unwrap_or_else(|| {
                st_config["attend_to_expansion_tokens"]
                    .as_bool()
                    .unwrap_or(false)
            });
        let final_query_length = builder
            .query_length
            .or_else(|| st_config["query_length"].as_u64().map(|v| v as usize));
        let final_document_length = builder
            .document_length
            .or_else(|| st_config["document_length"].as_u64().map(|v| v as usize));

        ColBERT::new(
            weights_bytes,
            dense_weights_bytes,
            tokenizer_bytes,
            config_bytes,
            dense_config_bytes,
            final_query_prefix,
            final_document_prefix,
            mask_token,
            final_do_query_expansion,
            final_attend_to_expansion_tokens,
            final_query_length,
            final_document_length,
            builder.batch_size,
            &device,
        )
    }
}
