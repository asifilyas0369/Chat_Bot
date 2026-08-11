from langchain_core.documents import Document

from src.generation.generator import generate_answer


documents = [
    Document(
        page_content=(
            "Machine learning is a field of study that "
            "focuses on algorithms that learn patterns "
            "from data."
        ),
        metadata={
            "source": "test_document.pdf",
            "page": 10,
        },
    ),
    Document(
        page_content=(
            "Machine learning systems can use training "
            "data to build predictive models."
        ),
        metadata={
            "source": "test_document.pdf",
            "page": 11,
        },
    ),
]


question = "What is machine learning?"


answer = generate_answer(
    question,
    documents,
)


print("\n==============================")
print("GROQ ANSWER")
print("==============================")

print(answer)