import json
import time

from pylate_rs import models

with open("benchmark/queries.json") as f:
    queries = json.load(f)

with open("benchmark/documents.json") as f:
    documents = json.load(f)

model_id = "lightonai/LateOn-Code-edge"

results = []

for mode, use_onnx in [("safetensors", False), ("onnx", True)]:
    print(f"\n--- {mode.upper()} ---")

    # Model loading
    start_load = time.time()
    model = models.ColBERT(
        model_name_or_path=model_id,
        device="cpu",
        use_onnx=use_onnx,
    )
    model_load_time = time.time() - start_load
    print(f"Model loaded in {model_load_time:.2f}s")

    # Encode queries
    start_query = time.time()
    queries_embeddings = model.encode(sentences=queries, is_query=True)
    query_time = time.time() - start_query
    qps = len(queries) / query_time

    # Encode documents
    start_doc = time.time()
    documents_embeddings = model.encode(sentences=documents, is_query=False)
    doc_time = time.time() - start_doc
    dps = len(documents) / doc_time

    results.append({
        "mode": mode,
        "load_time": model_load_time,
        "qps": qps,
        "dps": dps,
    })

print("\n\n--- Benchmark Summary ---")
header = f"{'Mode':<15} | {'Load Time (s)':<15} | {'QPS':>10} | {'DPS':>10}"
print(header)
print(f"{'-' * 15}-|-{'-' * 15}-|-{'-' * 10}-|-{'-' * 10}")

for res in results:
    row = (
        f"{res['mode']:<15} | "
        f"{res['load_time']:<15.2f} | "
        f"{res['qps']:>10.2f} | "
        f"{res['dps']:>10.2f}"
    )
    print(row)
