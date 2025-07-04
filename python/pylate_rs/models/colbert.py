from __future__ import annotations

import numpy as np
from pylate_rs.pylate_rs import (
    PyColBERT,
)


class ColBERT:
    """Python wrapper for the high-performance, Rust-based ColBERT model.

    Args:
    ----
        model_name_or_path:
            The identifier for a model on the Hugging Face Hub.
        device:
            The device to run the model on ("cpu" or "cuda"). Defaults to "cpu".
        query_length:
            The maximum length for queries. Defaults to 32.
        document_length:
            The maximum length for documents. Defaults to 180.
        batch_size:
            The batch size for encoding. Defaults to 32.
        attend_to_expansion_tokens:
            Whether to attend to expansion tokens in queries. Defaults to True.
        query_prefix:
            The prefix to add to queries. Defaults to query_prefix in the
            model config.
        document_prefix:
            The prefix to add to documents. Defaults to document_prefix in the
            model config.
        mask_token:
            The mask token used for padding queries. Defaults to "[MASK]".

    """

    def __init__(  # noqa: PLR0913
        self,
        model_name_or_path: str,
        device: str = "cpu",
        query_length: int = 32,
        document_length: int = 180,
        batch_size: int = 32,
        attend_to_expansion_tokens: bool = False,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        mask_token: str = "[MASK]",  # noqa: S107
    ):
        """Initialize and configures the ColBERT model."""
        self.model = PyColBERT.from_pretrained(
            repo_id=model_name_or_path,
            device=device,
            query_length=query_length,
            document_length=document_length,
            batch_size=batch_size,
            attend_to_expansion_tokens=attend_to_expansion_tokens,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            mask_token=mask_token,
        )

    def encode(
        self,
        sentences: list[str],
        is_query: bool,
        pool_factor: int = 1,
    ) -> np.ndarray:
        """Encode a list of sentences into embeddings.

        Args:
        ----
            sentences:
                A list of strings to encode.
            is_query:
                A flag indicating if the sentences are queries or documents.
            pool_factor:
                The factor by which to pool the embeddings. Defaults to 1.
                When set to 2, it will divide the number of tokens embedded by 2,
                effectively pooling the embeddings. You can set it to any integer
                value to control the pooling behavior. If set to 1, no pooling is
                applied.

        Returns:
        -------
            The resulting embeddings as a NumPy array.

        """
        return self.model.encode(
            sentences=sentences,
            is_query=is_query,
            pool_factor=pool_factor,
        )

    def similarity(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> list[list[float]]:
        """Calculate similarity scores between query and document embeddings.

        This method performs the late-interaction scoring that is central to ColBERT's
        design.

        Args:
        ----
            query_embeddings:
                A NumPy array of query embeddings.
            doc_embeddings:
                A NumPy array of document embeddings.

        Returns:
        -------
            A nested list containing the similarity scores.

        """
        return self.model.similarity(query_embeddings, doc_embeddings)
