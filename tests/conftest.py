"""Pytest configuration and fixtures for pylate vs pylate-rs comparison tests."""

from __future__ import annotations

import pytest

# Tolerance for numerical comparisons
RTOL = 1e-4
ATOL = 1e-5

# Models to test
MODELS = [
    "lightonai/GTE-ModernColBERT-v1",
    "lightonai/answerai-colbert-small-v1",
]


@pytest.fixture
def sample_queries() -> list[str]:
    """Sample queries for testing."""
    return [
        "What is the capital of France?",
        "How does machine learning work?",
    ]


@pytest.fixture
def sample_documents() -> list[str]:
    """Sample documents for testing."""
    return [
        "Paris is the capital of France and its largest city.",
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris.",
    ]


@pytest.fixture(params=MODELS)
def model_name(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture for model names."""
    return request.param
