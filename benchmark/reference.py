import json
import time

# Load queries and documents from JSON files
with open("benchmark/queries.json") as f:
    queries = json.load(f)

with open("benchmark/documents.json") as f:
    documents = json.load(f)

queries = [queries[0]]

whole_time = time.time()
from pylate_rs import models

# Detect available compute devices
devices = ["cpu"]


results = []

# Loop through each available device to run benchmarks
for device in devices:
    print(f"Testing on device: {device}...")

    # --- Model Loading ---
    start_load = time.time()
    model = models.ColBERT(
        model_name_or_path="lightonai/answerai-colbert-small-v1",
        device=device,
    )
    end_load = time.time()
    model_load_time = end_load - start_load

    # --- Query Encoding ---
    start_query = time.time()
    queries_embeddings = model.encode(
        sentences=queries,
        is_query=True,
    )
    end_query = time.time()
    query_time = end_query - start_query
    qps = len(queries) / query_time

    # --- Document Encoding ---
    start_doc = time.time()
    documents_embeddings = model.encode(
       sentences=documents,
        is_query=False,
        convert_to_tensor=False,
    )
    end_doc = time.time()
    doc_time = end_doc - start_doc
    dps = len(documents) / doc_time

    # Store results for the current device
    results.append(
        {"device": device, "load_time": model_load_time, "qps": qps, "dps": dps}
    )
    print(f"Finished testing on {device}.")


# --- Print Final Summary Table ---
print("\n--- Benchmark Summary ---")
header = f"{'Device':<10} | {'Model Loading Time (s)':<25} | {'QPS':>10} | {'DPS':>10}"
print(header)
print(f"{'-' * 10}-|-{'-' * 25}-|-{'-' * 10}-|-{'-' * 10}")

for res in results:
    row = (
        f"{res['device']:<10} | "
        f"{res['load_time']:<25.2f} | "
        f"{res['qps']:>10.2f} | "
        f"{res['dps']:>10.2f}"
    )
    print(row)


print(
    f"\nTotal time taken for the whole benchmark: {time.time() - whole_time:.2f} seconds"
)
