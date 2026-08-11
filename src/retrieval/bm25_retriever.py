from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Retriever:
    """
    Sparse keyword retriever using BM25.
    """

    def __init__(self, documents: list[Document]):
        self.documents = documents

        # Tokenize every document
        tokenized_documents = [
            document.page_content.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        """
        Search documents using BM25.
        """

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:k]:
            results.append(
                (
                    self.documents[index],
                    float(scores[index]),
                )
            )

        return results