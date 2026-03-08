import numpy as np
from pylate_rs import models

model_id = "lightonai/LateOn-Code"

queries = ["what is the capital of france", "how to sort a list in python"]
documents = ["paris is the capital of france", "use the sorted() function in python"]

# Load both models
print("Loading safetensors model...")
model_st = models.ColBERT(model_name_or_path=model_id, device="cpu", use_onnx=False)

print("Loading ONNX model...")
model_onnx = models.ColBERT(model_name_or_path=model_id, device="cpu", use_onnx=True)

# Encode with both
q_st = model_st.encode(sentences=queries, is_query=True)
q_onnx = model_onnx.encode(sentences=queries, is_query=True)

d_st = model_st.encode(sentences=documents, is_query=False)
d_onnx = model_onnx.encode(sentences=documents, is_query=False)

print(f"\nQuery embeddings shape:    safetensors={q_st.shape}  onnx={q_onnx.shape}")
print(f"Document embeddings shape: safetensors={d_st.shape}  onnx={d_onnx.shape}")

# Compare embeddings
print("\n--- Embeddings Comparison ---")
q_max_diff = np.max(np.abs(q_st - q_onnx))
q_mean_diff = np.mean(np.abs(q_st - q_onnx))
d_max_diff = np.max(np.abs(d_st - d_onnx))
d_mean_diff = np.mean(np.abs(d_st - d_onnx))

print(f"Queries  - max diff: {q_max_diff:.6e}  mean diff: {q_mean_diff:.6e}")
print(f"Documents - max diff: {d_max_diff:.6e}  mean diff: {d_mean_diff:.6e}")

# Compare similarity scores
sim_st = model_st.similarity(q_st, d_st)
sim_onnx = model_onnx.similarity(q_onnx, d_onnx)

print("\n--- Similarity Scores ---")
for i, q in enumerate(queries):
    print(f"\n  {q}:")
    for j, d in enumerate(documents):
        diff = abs(sim_st[i][j] - sim_onnx[i][j])
        print(f"    {d}")
        print(f"      safetensors: {sim_st[i][j]:.6f}  onnx: {sim_onnx[i][j]:.6f}  diff: {diff:.6e}")
