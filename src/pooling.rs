use anyhow::anyhow;
use candle_core::{Device, Tensor};
use candle_transformers::models::deepseek2::NonZeroOp;
use kodama::{linkage, Method};
use std::collections::HashMap;

/// A Disjoint Set Union (DSU) data structure using a HashMap to handle generic cluster labels.
struct Dsu {
    parent: HashMap<usize, usize>,
}

impl Dsu {
    /// Creates a new, empty DSU.
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
        }
    }

    /// Finds the representative (root) of the set containing `i`.
    fn find(&mut self, i: usize) -> usize {
        let parent_i = *self.parent.entry(i).or_insert(i);

        if parent_i == i {
            return i;
        }

        let root = self.find(parent_i);
        self.parent.insert(i, root);
        root
    }

    /// Merges the sets containing `i` and `j`.
    fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            self.parent.insert(root_i, root_j);
        }
    }
}

/// Performs hierarchical pooling on a batch of document embeddings.
pub fn hierarchical_pooling(
    documents_embeddings: &Tensor,
    pool_factor: usize,
) -> anyhow::Result<Tensor> {
    if pool_factor <= 1 {
        return Ok(documents_embeddings.clone());
    }

    if documents_embeddings.dims().len() != 3 {
        return Err(anyhow!(
            "Input tensor must have 3 dimensions [batch_size, n_tokens, embedding_dim], but got {} dimensions.",
            documents_embeddings.dims().len()
        ));
    }

    let device = documents_embeddings.device();
    let documents_embeddings = if !device.is_cpu() {
        documents_embeddings.to_device(&Device::Cpu)?
    } else {
        documents_embeddings.clone()
    };

    let device = documents_embeddings.device();
    let mut all_pooled_embeddings: Vec<Tensor> = Vec::new();
    let batch_size = documents_embeddings.dim(0)?;

    for i in 0..batch_size {
        let document_embeddings = documents_embeddings.narrow(0, i, 1)?.squeeze(0)?;
        let n_tokens = document_embeddings.dim(0)?;

        if 1 >= n_tokens {
            all_pooled_embeddings.push(document_embeddings.clone());
            continue;
        }

        let protected_embeddings = document_embeddings.narrow(0, 0, 1)?;
        let embeddings_to_pool = document_embeddings.narrow(0, 1, n_tokens - 1)?;
        let num_embeddings_to_pool = embeddings_to_pool.dim(0)?;

        if num_embeddings_to_pool <= 1 {
            let final_embeddings = Tensor::cat(&[&protected_embeddings, &embeddings_to_pool], 0)?;
            all_pooled_embeddings.push(final_embeddings);
            continue;
        }

        let cosine_similarities = embeddings_to_pool.matmul(&embeddings_to_pool.t()?)?;
        let distance_matrix_tensor = (1.0 - cosine_similarities)?;

        let mut condensed_distances: Vec<f64> = Vec::new();
        for row in 0..num_embeddings_to_pool - 1 {
            for col in row + 1..num_embeddings_to_pool {
                let dist = distance_matrix_tensor
                    .get(row)?
                    .get(col)?
                    .to_scalar::<f32>()? as f64;
                condensed_distances.push(dist);
            }
        }

        let dend = linkage(
            &mut condensed_distances,
            num_embeddings_to_pool,
            Method::Ward,
        );

        let num_clusters = (num_embeddings_to_pool / pool_factor).max(1);

        if num_clusters >= num_embeddings_to_pool {
            let final_embeddings = Tensor::cat(&[&protected_embeddings, &embeddings_to_pool], 0)?;
            all_pooled_embeddings.push(final_embeddings);
            continue;
        }

        let mut dsu = Dsu::new();
        let num_merges = num_embeddings_to_pool - num_clusters;

        for step in dend.steps().iter().take(num_merges) {
            dsu.union(step.cluster1, step.cluster2);
        }

        let mut root_to_label = HashMap::new();
        let mut labels = Vec::with_capacity(num_embeddings_to_pool);
        for i in 0..num_embeddings_to_pool {
            let root = dsu.find(i);
            let next_label = root_to_label.len();
            let label = *root_to_label.entry(root).or_insert(next_label);
            labels.push(label as u32);
        }
        let labels_tensor = Tensor::new(labels.as_slice(), device)?;

        let mut pooled_document_embeddings: Vec<Tensor> = Vec::with_capacity(num_clusters);
        for cluster_id in 0..num_clusters {
            let mask = labels_tensor.eq(cluster_id as u32)?;
            let cluster_indices = mask.nonzero()?.squeeze(1)?;

            if cluster_indices.dim(0)? > 0 {
                let cluster_embeddings = embeddings_to_pool.index_select(&cluster_indices, 0)?;
                pooled_document_embeddings.push(cluster_embeddings.mean(0)?);
            }
        }

        let mut final_embeddings_list = pooled_document_embeddings;
        for j in 0..1 {
            final_embeddings_list.push(protected_embeddings.get(j)?);
        }

        let final_doc_tensor = Tensor::stack(&final_embeddings_list, 0)?;
        all_pooled_embeddings.push(final_doc_tensor);
    }

    Ok(Tensor::stack(&all_pooled_embeddings, 0)?)
}
