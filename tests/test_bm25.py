from langchain_community.document_loaders import PyPDFLoader

from src.chunking.chunker import chunk_documents
from src.retrieval.bm25_retriever import BM25Retriever


PDF_PATH = "data/raw/Machine learning.pdf"


# -----------------------------
# Load PDF
# -----------------------------

print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Pages loaded: {len(documents)}")


# -----------------------------
# Create chunks
# -----------------------------

print("\nCreating chunks...")

chunks = chunk_documents(
    documents,
    strategy="recursive",
    chunk_size=1000,
    chunk_overlap=200,
)

print(f"Total chunks: {len(chunks)}")


# -----------------------------
# Create BM25 index
# -----------------------------

print("\nCreating BM25 index...")

retriever = BM25Retriever(chunks)

print("BM25 index created!")


# -----------------------------
# Search
# -----------------------------

query = "machine learning system"

results = retriever.search(
    query,
    k=5,
)


# -----------------------------
# Display results
# -----------------------------

print("\n==============================")
print("BM25 SEARCH RESULTS")
print("==============================")


for i, (document, score) in enumerate(results):

    print(f"\nResult {i + 1}")
    print(f"BM25 score: {score:.4f}")

    print("\nText:")
    print(document.page_content[:500])

    print("\nMetadata:")
    print(document.metadata)