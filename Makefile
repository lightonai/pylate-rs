test:
	cargo test --features metal,accelerate -- --nocapture

install-wasm:
	cargo install wasm-pack

build-wasm:
	rm -rf docs/pkg
	wasm-pack build . --target web --out-dir ./docs/pkg -- --features wasm --no-default-features
	python3 -m http.server --directory docs

run-rust:
	cargo build --release --features metal,accelerate --bin benchmark
	./target/release/benchmark

run-python:
	cargo clean
	pip uninstall -y pylate-rs
	python generate_configs.py metal
	pip install .
	python benchmark/python.py

lint:
	cargo clean
	uv run --extra dev pre-commit run --files python/**/**/**.py


run:
	cargo clean
	pip uninstall pylate-rs
	pip install .
	python test.py