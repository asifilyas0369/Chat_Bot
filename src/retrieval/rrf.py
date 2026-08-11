from langchain_core.documents import Document


def reciprocal_rank_fusion(
    dense_results: list[tuple[Document, float]],
    sparse_results: list[tuple[Document, float]],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    k: int = 60,
) -> list[tuple[Document, float]]:
    """
    Combine dense and sparse retrieval results
    using Reciprocal Rank Fusion (RRF).
    """

    scores = {}
    documents = {}

    # -----------------------------
    # Dense results
    # -----------------------------

    for rank, (document, _) in enumerate(
        dense_results,
        start=1,
    ):
        document_id = document.metadata.get(
            "chunk_id",
            document.page_content,
        )

        rrf_score = dense_weight / (k + rank)

        scores[document_id] = (
            scores.get(document_id, 0) + rrf_score
        )

        documents[document_id] = document

    # -----------------------------
    # Sparse results
    # -----------------------------

    for rank, (document, _) in enumerate(
        sparse_results,
        start=1,
    ):
        document_id = document.metadata.get(
            "chunk_id",
            document.page_content,
        )

        rrf_score = sparse_weight / (k + rank)

        scores[document_id] = (
            scores.get(document_id, 0) + rrf_score
        )

        documents[document_id] = document

    # -----------------------------
    # Sort by RRF score
    # -----------------------------

    ranked_results = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (documents[document_id], score)
        for document_id, score in ranked_results
    ]