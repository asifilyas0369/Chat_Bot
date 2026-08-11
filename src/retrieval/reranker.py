from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker:
    """
    Cross-encoder reranker for retrieved documents.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):
        print(f"Loading reranker: {model_name}")

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Rerank documents based on query-document relevance.
        """

        pairs = [
            [query, document.page_content]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked_results = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            (document, float(score))
            for document, score in ranked_results[:top_k]
        ]