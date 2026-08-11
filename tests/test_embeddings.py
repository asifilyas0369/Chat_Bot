from src.embeddings.embedder import create_embedding_model


embedding_model = create_embedding_model()

text = "Machine learning systems require reliable data pipelines."

embedding = embedding_model.embed_query(text)

print("Embedding created successfully!")
print(f"Dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")