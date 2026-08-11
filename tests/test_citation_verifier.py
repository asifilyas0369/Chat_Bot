from langchain_core.documents import Document
from dotenv import load_dotenv

from src.generation.generator import create_llm
from src.generation.citation_verifier import verify_citation


load_dotenv()


# Test source
source = Document(
    page_content=(
        "Machine learning is a field of study that "
        "focuses on algorithms that learn patterns "
        "from data."
    ),
    metadata={
        "source": "test_document.pdf",
        "page": 10,
    },
)


# Claim that should be supported
claim = (
    "Machine learning algorithms can learn patterns "
    "from data."
)


print("Creating Groq LLM...")

llm = create_llm()


print("Testing citation verification...")

result = verify_citation(
    claim=claim,
    source=source,
    llm=llm,
)


print("\n==============================")
print("CITATION VERIFICATION")
print("==============================")

print(f"Claim: {claim}")
print(f"Verified: {result}")