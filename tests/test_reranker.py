from langchain_community.document_loaders import PyPDFLoader

from src.chunking.chunker import chunk_documents
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import create_vector_store
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker


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
# 3. Test collection
# --------------------------------

test_chunks = chunks[:100]

print(f"\nUsing {len(test_chunks)} chunks.")


# --------------------------------
# 4. Dense retrieval
# --------------------------------

print("\nCreating ChromaDB...")

vector_store = create_vector_store(test_chunks)


# --------------------------------
# 5. BM25 retrieval
# --------------------------------

print("Creating BM25...")

bm25 = BM25Retriever(test_chunks)


# --------------------------------
# 6. Query
# --------------------------------

query = "What is machine learning?"


dense_results = (
    vector_store.similarity_search_with_score(
        query,
        k=10,
    )
)

sparse_results = bm25.search(
    query,
    k=10,
)


# --------------------------------
# 7. RRF
# --------------------------------

dense_results = [
    (document, float(score))
    for document, score in dense_results
]

fused_results = reciprocal_rank_fusion(
    dense_results=dense_results,
    sparse_results=sparse_results,
    dense_weight=0.7,
    sparse_weight=0.3,
)


# --------------------------------
# 8. Take RRF candidates
# --------------------------------

candidates = [
    document
    for document, score in fused_results[:20]
]

print(f"\nRRF candidates: {len(candidates)}")


# --------------------------------
# 9. Reranking
# --------------------------------

print("\nRunning reranker...")

reranker = Reranker()

reranked_results = reranker.rerank(
    query=query,
    documents=candidates,
    top_k=5,
)


# --------------------------------
# 10. Display
# --------------------------------

print("\n==============================")
print("RERANKED RESULTS")
print("==============================")


for rank, (document, score) in enumerate(
    reranked_results,
    start=1,
):

    print(f"\nRank {rank}")
    print(f"Reranker score: {score:.4f}")

    print("\nText:")
    print(document.page_content[:500])

    print("\nMetadata:")
    print(document.metadata)