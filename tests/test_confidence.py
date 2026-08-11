from src.evaluation.confidence import calculate_confidence


result = calculate_confidence(
    retrieval_confidence=0.88,
    citation_coverage=1.0,
    completeness=0.90,
)


print("\n==============================")
print("CONFIDENCE SCORE")
print("==============================")

print(f"Retrieval confidence: {result['retrieval_confidence']}")
print(f"Citation coverage:    {result['citation_coverage']}")
print(f"Completeness:          {result['completeness']}")
print(f"Overall confidence:    {result['overall_confidence']}")