def calculate_confidence(
    retrieval_confidence: float,
    citation_coverage: float,
    completeness: float,
) -> dict:
    """
    Calculate the overall confidence of a RAG answer.
    """

    overall = (
        0.4 * retrieval_confidence
        + 0.4 * citation_coverage
        + 0.2 * completeness
    )

    return {
        "retrieval_confidence": round(
            retrieval_confidence, 3
        ),
        "citation_coverage": round(
            citation_coverage, 3
        ),
        "completeness": round(
            completeness, 3
        ),
        "overall_confidence": round(
            overall, 3
        ),
    }