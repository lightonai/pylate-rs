# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pylate-rs is a high-performance inference engine for PyLate late-interaction models (ColBERT architecture), written in Rust with bindings for Python, JavaScript/WebAssembly, and native Rust. Built on the Candle ML framework for minimal footprint.

## Build Commands

### Python (via Maturin)
```bash
# Install from source (auto-detects hardware)
pip install .

# Build wheel for specific variant
python generate_configs.py metal  # Options: metal, accelerate, mkl, cuda
maturin build --release --out dist
```

### Rust
```bash
# Build with hardware acceleration (macOS)
cargo build --release --features metal,accelerate

# Build benchmark binary
cargo build --release --features metal,accelerate --bin benchmark
```

### WebAssembly
```bash
cargo install wasm-pack
wasm-pack build . --target web --out-dir ./docs/pkg -- --features wasm --no-default-features
```

## Test Commands

```bash
# Rust tests
cargo test --features metal,accelerate -- --nocapture

# Python tests
pytest tests/

# Integration test comparing with PyLate
python test.py
```

## Lint/Format

```bash
make lint
# or directly:
uv run --extra dev pre-commit run --files python/**/**/**.py
```

## Benchmark Commands

```bash
# Rust benchmark
make run-rust

# Python benchmark (rebuilds from source)
make run-python
```

## Architecture

### Core Components (src/)
- **model.rs** - Main ColBERT model: encoding, similarity computation, tensor operations
- **builder.rs** - ColbertBuilder for model configuration and HuggingFace Hub loading
- **modernbert.rs** - ModernBert transformer implementation (local/global attention)
- **pooling.rs** - Hierarchical pooling algorithm for reducing embedding dimensions
- **python.rs** - PyO3 bindings exposing PyColbERT class
- **wasm.rs** - wasm-bindgen bindings for browser/Node.js

### Python Wrapper (python/pylate_rs/)
- **models/colbert.py** - High-level Python API wrapping the Rust implementation

### Data Flow
1. Model loading: ColbertBuilder → HF Hub download → parse configs → load weights → ColBERT
2. Encoding: sentences → tokenize → forward pass → linear projection → embeddings
3. Similarity: query embeddings × document embeddings → ColBERT-style late interaction → scores

### Hardware Backend Features
- `metal` - Apple GPU (M1/M2/M3)
- `cuda` - NVIDIA GPU
- `mkl` - Intel CPU optimization
- `accelerate` - Apple CPU optimization
- Default - Standard CPU

### Key Dependencies
- candle-core/nn/transformers (v0.9.1) - HuggingFace's Rust ML framework
- tokenizers (v0.21.1) - HuggingFace tokenizers
- pyo3 (v0.25.1) - Python bindings
- kodama (v0.3.0) - Hierarchical clustering for pooling

## Model Compatibility

Models must be saved in PyLate format. Convert existing models using:
```python
from pylate import models
model = models.ColBERT(model_name_or_path="your-model")
model.push_to_hub("YourUsername/YourModelName")
```

Compatible models: https://huggingface.co/collections/lightonai/pylate-6862b571946fe88330d65264
