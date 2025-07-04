test:
	cargo test --features metal,accelerate -- --nocapture

install-wasm:
	cargo install wasm-pack

build-wasm:
	rm -rf public/pkg
	wasm-pack build . --target web --out-dir ./doc/pkg -- --features wasm --no-default-features
	python3 -m http.server --directory doc

run-rust:
	cargo build --release --features metal,accelerate --bin benchmark
	./target/release/benchmark

run-python:
	cargo clean
	pip uninstall -y pylate-rs
	pip install ".[metal]"
	python benchmark/python.py

lint:
	cargo clean
	uv run --extra dev pre-commit run --files python/**/**/**.py


