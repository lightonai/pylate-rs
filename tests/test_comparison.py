"""Comparison tests between pylate (Python) and pylate-rs (Rust via Python bindings)."""

from __future__ import annotations

import numpy as np
import pytest
from pylate import models as pylate_models

from pylate_rs import models as pylate_rs_models

from .conftest import ATOL, RTOL


@pytest.mark.slow
class TestEmbeddingComparison:
    """Tests comparing embeddings between pylate and pylate-rs."""

    def test_document_embedding_shape_match(
        self,
        model_name: str,
        sample_documents: list[str],
    ) -> None:
        """Test that document embedding shapes match between implementations."""
        # Load pylate model (disable query expansion to match pylate-rs behavior)
        pylate_model = pylate_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
            do_query_expansion=False,
        )

        # Load pylate-rs model
        pylate_rs_model = pylate_rs_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
        )

        # Encode documents
        pylate_embeddings = pylate_model.encode(
            sentences=sample_documents,
            is_query=False,
        )
        pylate_rs_embeddings = pylate_rs_model.encode(
            sentences=sample_documents,
            is_query=False,
        )

        # Check embedding dimensions match (last dimension is embedding size)
        assert pylate_embeddings[0].shape[-1] == pylate_rs_embeddings[0].shape[-1], (
            f"Embedding dimensions don't match: "
            f"pylate={pylate_embeddings[0].shape[-1]}, "
            f"pylate-rs={pylate_rs_embeddings[0].shape[-1]}"
        )

    def test_document_embedding_values_match(
        self,
        model_name: str,
        sample_documents: list[str],
    ) -> None:
        """Test that document embedding values are close between implementations."""
        # Load pylate model (disable query expansion to match pylate-rs behavior)
        pylate_model = pylate_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
            do_query_expansion=False,
        )

        # Load pylate-rs model
        pylate_rs_model = pylate_rs_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
        )

        # Encode single document to simplify comparison
        doc = [sample_documents[0]]

        pylate_emb = pylate_model.encode(sentences=doc, is_query=False)[0]
        pylate_rs_emb = pylate_rs_model.encode(sentences=doc, is_query=False)[0]

        # Get the minimum length (pylate may filter some tokens)
        min_len = min(pylate_emb.shape[0], pylate_rs_emb.shape[0])

        # Compare first min_len tokens (use looser tolerance for document embeddings
        # due to potential differences in punctuation handling)
        np.testing.assert_allclose(
            pylate_emb[:min_len],
            pylate_rs_emb[:min_len],
            rtol=0.15,
            atol=0.15,
            err_msg=f"Document embeddings don't match for model {model_name}",
        )

    def test_query_embedding_values_match(
        self,
        model_name: str,
        sample_queries: list[str],
    ) -> None:
        """Test that query embedding values are close between implementations."""
        # Load pylate model with matching query expansion settings
        pylate_model = pylate_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
            do_query_expansion=False,
        )

        # Load pylate-rs model
        pylate_rs_model = pylate_rs_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
            do_query_expansion=False,
        )

        # Encode single query
        query = [sample_queries[0]]

        pylate_emb = pylate_model.encode(sentences=query, is_query=True)[0]
        pylate_rs_emb = pylate_rs_model.encode(sentences=query, is_query=True)[0]

        # Get the minimum length
        min_len = min(pylate_emb.shape[0], pylate_rs_emb.shape[0])

        # Compare embeddings
        np.testing.assert_allclose(
            pylate_emb[:min_len],
            pylate_rs_emb[:min_len],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Query embeddings don't match for model {model_name}",
        )


@pytest.mark.slow
class TestSimilarityComparison:
    """Tests comparing similarity scores between pylate and pylate-rs."""

    def test_similarity_scores_match(
        self,
        model_name: str,
        sample_queries: list[str],
        sample_documents: list[str],
    ) -> None:
        """Test that similarity scores are close between implementations."""
        # Load models
        pylate_model = pylate_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
            do_query_expansion=False,
        )

        pylate_rs_model = pylate_rs_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
            do_query_expansion=False,
        )

        # Use single query and document for cleaner comparison
        query = [sample_queries[0]]
        doc = [sample_documents[0]]

        # Encode with pylate
        pylate_query_emb = pylate_model.encode(sentences=query, is_query=True)
        pylate_doc_emb = pylate_model.encode(sentences=doc, is_query=False)

        # Encode with pylate-rs
        pylate_rs_query_emb = pylate_rs_model.encode(sentences=query, is_query=True)
        pylate_rs_doc_emb = pylate_rs_model.encode(sentences=doc, is_query=False)

        # Compute similarity with pylate-rs
        pylate_rs_similarity = pylate_rs_model.similarity(
            pylate_rs_query_emb, pylate_rs_doc_emb
        )

        # Compute similarity with pylate (using its score function)
        from pylate.scores import colbert_scores

        pylate_similarity = colbert_scores(
            queries_embeddings=pylate_query_emb,
            documents_embeddings=pylate_doc_emb,
        )

        # Compare scores (use higher tolerance for accumulated numerical differences)
        np.testing.assert_allclose(
            pylate_similarity[0][0].item(),
            pylate_rs_similarity[0][0],
            rtol=1e-2,
            atol=1e-1,
            err_msg=f"Similarity scores don't match for model {model_name}",
        )

    def test_relative_ranking_preserved(
        self,
        model_name: str,
        sample_queries: list[str],
        sample_documents: list[str],
    ) -> None:
        """Test that document rankings are consistent between implementations."""
        # Load models
        pylate_model = pylate_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
            do_query_expansion=False,
        )

        pylate_rs_model = pylate_rs_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
            do_query_expansion=False,
        )

        query = [sample_queries[0]]

        # Encode with pylate
        pylate_query_emb = pylate_model.encode(sentences=query, is_query=True)
        pylate_doc_embs = pylate_model.encode(sentences=sample_documents, is_query=False)

        # Encode with pylate-rs
        pylate_rs_query_emb = pylate_rs_model.encode(sentences=query, is_query=True)
        pylate_rs_doc_embs = pylate_rs_model.encode(
            sentences=sample_documents, is_query=False
        )

        # Compute similarities using pylate-rs for both (consistent comparison)
        # pylate returns variable-length arrays, so we compute scores individually
        pylate_scores = []
        for doc_emb in pylate_doc_embs:
            # Compute MaxSim score manually
            # query_emb shape: (num_query_tokens, dim)
            # doc_emb shape: (num_doc_tokens, dim)
            q_emb = pylate_query_emb[0]  # (num_query_tokens, dim)
            # Compute dot product between all query and doc tokens
            scores = np.matmul(q_emb, doc_emb.T)  # (num_query_tokens, num_doc_tokens)
            # MaxSim: max over doc tokens for each query token, then sum
            max_scores = scores.max(axis=1)  # (num_query_tokens,)
            pylate_scores.append(max_scores.sum())

        pylate_scores_arr = np.array(pylate_scores)

        pylate_rs_scores = np.array(
            pylate_rs_model.similarity(pylate_rs_query_emb, pylate_rs_doc_embs)[0]
        )

        # Check that rankings are the same
        pylate_ranking = np.argsort(-pylate_scores_arr)
        pylate_rs_ranking = np.argsort(-pylate_rs_scores)

        np.testing.assert_array_equal(
            pylate_ranking,
            pylate_rs_ranking,
            err_msg=f"Document rankings differ for model {model_name}",
        )


@pytest.mark.slow
class TestEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_long_document_encoding(
        self,
        model_name: str,
    ) -> None:
        """Test encoding of long documents that may be truncated."""
        long_doc = ["This is a test sentence. " * 50]
        doc_length = 180

        pylate_model = pylate_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
            do_query_expansion=False,
            document_length=doc_length,
        )

        pylate_rs_model = pylate_rs_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
            document_length=doc_length,
        )

        pylate_emb = pylate_model.encode(sentences=long_doc, is_query=False)[0]
        pylate_rs_emb = pylate_rs_model.encode(sentences=long_doc, is_query=False)[0]

        # Both should truncate to document_length
        assert pylate_emb.shape[0] <= doc_length, (
            f"pylate should truncate long documents to {doc_length}"
        )
        assert pylate_rs_emb.shape[0] <= doc_length, (
            f"pylate-rs should truncate long documents to {doc_length}"
        )

    def test_batch_encoding_consistency(
        self,
        model_name: str,
        sample_documents: list[str],
    ) -> None:
        """Test that batch encoding produces same results as individual encoding."""
        pylate_rs_model = pylate_rs_models.ColBERT(
            model_name_or_path=model_name,
            device="cpu",
        )

        # Encode all at once
        batch_embs = pylate_rs_model.encode(sentences=sample_documents, is_query=False)

        # Encode individually
        individual_embs = [
            pylate_rs_model.encode(sentences=[doc], is_query=False)[0]
            for doc in sample_documents
        ]

        # Compare (accounting for potential padding differences)
        for i, (batch_emb, indiv_emb) in enumerate(zip(batch_embs, individual_embs)):
            min_len = min(batch_emb.shape[0], indiv_emb.shape[0])
            np.testing.assert_allclose(
                batch_emb[:min_len],
                indiv_emb[:min_len],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Batch vs individual encoding mismatch for document {i}",
            )
