from langchain_community.document_loaders import PyPDFLoader

from src.chunking.chunker import chunk_documents
from src.retrieval.vector_store import create_vector_store


PDF_PATH = "data/raw/machine learning.pdf"


print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Pages loaded: {len(documents)}")


print("\nCreating chunks...")

chunks = chunk_documents(
    documents,
    strategy="recursive",
    chunk_size=1000,
    chunk_overlap=200,
)

print(f"Total chunks: {len(chunks)}")


# For the first test, use only 100 chunks.
test_chunks = chunks[:100]

print(f"\nAdding {len(test_chunks)} chunks to ChromaDB...")

vector_store = create_vector_store(test_chunks)

print("\nChromaDB created successfully!")


# Test similarity search
query = "What is machine learning?"

results = vector_store.similarity_search(
    query,
    k=5,
)


print("\n==============================")
print("SEARCH RESULTS")
print("==============================")


for i, result in enumerate(results):

    print(f"\nResult {i + 1}")

    print("------------------------------")

    print(result.page_content[:500])

    print("\nMetadata:")
    print(result.metadata)