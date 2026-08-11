from langchain_community.document_loaders import PyPDFLoader

from src.chunking.chunker import chunk_documents
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import create_vector_store
from src.retrieval.rrf import reciprocal_rank_fusion


PDF_PATH = "data/raw/machine learning.pdf"


# --------------------------------
# 1. Load PDF
# --------------------------------

print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Pages loaded: {len(documents)}")


# --------------------------------
# 2. Create chunks
# --------------------------------

print("\nCreating chunks...")

chunks = chunk_documents(
    documents,
    strategy="recursive",
    chunk_size=1000,
    chunk_overlap=200,
)

print(f"Total chunks: {len(chunks)}")


# --------------------------------
# 3. Use a small test collection
# --------------------------------

test_chunks = chunks[:100]

print(f"\nUsing {len(test_chunks)} chunks for testing.")


# --------------------------------
# 4. Create dense retriever
# --------------------------------

print("\nCreating ChromaDB...")

vector_store = create_vector_store(test_chunks)


# --------------------------------
# 5. Create BM25 retriever
# --------------------------------

print("Creating BM25...")

bm25 = BM25Retriever(test_chunks)


# --------------------------------
# 6. Search
# --------------------------------

query = "machine learning system"

print("\nSearching...")


dense_results = vector_store.similarity_search_with_score(
    query,
    k=10,
)

sparse_results = bm25.search(
    query,
    k=10,
)


# --------------------------------
# 7. Convert Chroma results
# --------------------------------

dense_results = [
    (document, float(score))
    for document, score in dense_results
]


# --------------------------------
# 8. RRF Fusion
# --------------------------------

print("\nRunning RRF...")

fused_results = reciprocal_rank_fusion(
    dense_results=dense_results,
    sparse_results=sparse_results,
    dense_weight=0.7,
    sparse_weight=0.3,
)


# --------------------------------
# 9. Display results
# --------------------------------

print("\n==============================")
print("RRF RESULTS")
print("==============================")


for rank, (document, score) in enumerate(
    fused_results[:10],
    start=1,
):

    print(f"\nRank {rank}")
    print(f"RRF Score: {score:.6f}")

    print("\nText:")
    print(document.page_content[:300])

    print("\nMetadata:")
    print(document.metadata)