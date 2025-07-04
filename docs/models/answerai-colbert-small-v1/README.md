---
tags:
- ColBERT
- PyLate
- sentence-transformers
- sentence-similarity
- feature-extraction
base_model: answerdotai/answerai-colbert-small-v1
pipeline_tag: sentence-similarity
library_name: PyLate
---

# PyLate model based on answerdotai/answerai-colbert-small-v1

This is a [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) model converted to [PyLate](https://github.com/lightonai/pylate) format. It maps sentences & paragraphs to sequences of 96-dimensional dense vectors and can be used for semantic textual similarity using the MaxSim operator.

## Model Details

### Model Description
- **Model Type:** PyLate model
- **Base model:** [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) <!-- at revision be1703c55532145a844da800eea4c9a692d7e267 -->
- **Document Length:** 300 tokens
- **Query Length:** 32 tokens
- **Output Dimensionality:** 96 tokens
- **Similarity Function:** MaxSim
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [PyLate Documentation](https://lightonai.github.io/pylate/)
- **Repository:** [PyLate on GitHub](https://github.com/lightonai/pylate)
- **Hugging Face:** [PyLate models on Hugging Face](https://huggingface.co/models?library=PyLate)

### Full Model Architecture

```
ColBERT(
  (0): Transformer({'max_seq_length': 299, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Dense({'in_features': 384, 'out_features': 96, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
)
```

## Usage
First install the PyLate library:

```bash
pip install -U pylate
```

### Retrieval

PyLate provides a streamlined interface to index and retrieve documents using ColBERT models. The index leverages the Voyager HNSW index to efficiently handle document embeddings and enable fast retrieval.

#### Indexing documents

First, load the ColBERT model and initialize the Voyager index, then encode and index your documents:

```python
from pylate import indexes, models, retrieve

# Step 1: Load the ColBERT model
model = models.ColBERT(
    model_name_or_path=lightonai/answerai-colbert-small-v1,
)

# Step 2: Initialize the Voyager index
index = indexes.Voyager(
    index_folder="pylate-index",
    index_name="index",
    override=True,  # This overwrites the existing index if any
)

# Step 3: Encode the documents
documents_ids = ["1", "2", "3"]
documents = ["document 1 text", "document 2 text", "document 3 text"]

documents_embeddings = model.encode(
    documents,
    batch_size=32,
    is_query=False,  # Ensure that it is set to False to indicate that these are documents, not queries
    show_progress_bar=True,
)

# Step 4: Add document embeddings to the index by providing embeddings and corresponding ids
index.add_documents(
    documents_ids=documents_ids,
    documents_embeddings=documents_embeddings,
)
```

Note that you do not have to recreate the index and encode the documents every time. Once you have created an index and added the documents, you can re-use the index later by loading it:

```python
# To load an index, simply instantiate it with the correct folder/name and without overriding it
index = indexes.Voyager(
    index_folder="pylate-index",
    index_name="index",
)
```

#### Retrieving top-k documents for queries

Once the documents are indexed, you can retrieve the top-k most relevant documents for a given set of queries.
To do so, initialize the ColBERT retriever with the index you want to search in, encode the queries and then retrieve the top-k documents to get the top matches ids and relevance scores:

```python
# Step 1: Initialize the ColBERT retriever
retriever = retrieve.ColBERT(index=index)

# Step 2: Encode the queries
queries_embeddings = model.encode(
    ["query for document 3", "query for document 1"],
    batch_size=32,
    is_query=True,  #  # Ensure that it is set to False to indicate that these are queries
    show_progress_bar=True,
)

# Step 3: Retrieve top-k documents
scores = retriever.retrieve(
    queries_embeddings=queries_embeddings,
    k=10,  # Retrieve the top 10 matches for each query
)
```

### Reranking
If you only want to use the ColBERT model to perform reranking on top of your first-stage retrieval pipeline without building an index, you can simply use rank function and pass the queries and documents to rerank:

```python
from pylate import rank, models

queries = [
    "query A",
    "query B",
]

documents = [
    ["document A", "document B"],
    ["document 1", "document C", "document B"],
]

documents_ids = [
    [1, 2],
    [1, 3, 2],
]

model = models.ColBERT(
    model_name_or_path=lightonai/answerai-colbert-small-v1,
)

queries_embeddings = model.encode(
    queries,
    is_query=True,
)

documents_embeddings = model.encode(
    documents,
    is_query=False,
)

reranked_documents = rank.rerank(
    documents_ids=documents_ids,
    queries_embeddings=queries_embeddings,
    documents_embeddings=documents_embeddings,
)
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Framework Versions
- Python: 3.12.0
- Sentence Transformers: 4.0.2
- PyLate: 1.2.0
- Transformers: 4.48.2
- PyTorch: 2.6.0
- Accelerate: 1.6.0
- Datasets: 3.5.0
- Tokenizers: 0.21.1


## Citation

### BibTeX

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->