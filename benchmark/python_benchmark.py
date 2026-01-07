import json
import time

import torch
from pylate_rs import models

with open("benchmark/queries.json") as f:
    queries = json.load(f)

with open("benchmark/documents.json") as f:
    documents = json.load(f)


queries = queries
documents = documents

devices = (
    ["cuda"]
    if torch.cuda.is_available()
    else [] + ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
)

results = []


for device in devices:
    print(f"Testing on device: {device}...")

    start_load = time.time()
    model = models.ColBERT(
        model_name_or_path="lightonai/answerai-colbert-small-v1",
        device=device,
    )
    end_load = time.time()
    model_load_time = end_load - start_load

    start_query = time.time()
    queries_embeddings = model.encode(
        sentences=queries,
        is_query=True,
    )
    end_query = time.time()
    query_time = end_query - start_query
    qps = len(queries) / query_time

    start_doc = time.time()
    documents_embeddings = model.encode(
        sentences=documents,
        is_query=False,
    )
    end_doc = time.time()
    doc_time = end_doc - start_doc
    dps = len(documents) / doc_time

    results.append(
        {"device": device, "load_time": model_load_time, "qps": qps, "dps": dps}
    )
    print(f"Finished testing on {device}.")


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
