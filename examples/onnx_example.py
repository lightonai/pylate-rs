from pylate_rs.models import ColBERT

# Load the model using ONNX
model = ColBERT(
    model_name_or_path="lightonai/LateOn-Code-edge",
    use_onnx=True,
)

# Queries
queries = [
    "query A",
    "query B",
]

# Documents
documents = [
    "document A",
    "document B",
    "document 1",
    "document C",
]

# Encode queries and documents
queries_embeddings = model.encode(queries, is_query=True)
documents_embeddings = model.encode(documents, is_query=False)

print("Queries embeddings shape:", queries_embeddings.shape)
print("Documents embeddings shape:", documents_embeddings.shape)

# Compute similarity scores
scores = model.similarity(queries_embeddings, documents_embeddings)

print("\nSimilarity scores:")
for i, query in enumerate(queries):
    print(f"\n  {query}:")
    for j, doc in enumerate(documents):
        print(f"    {doc}: {scores[i][j]:.4f}")
