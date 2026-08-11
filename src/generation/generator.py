import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document


load_dotenv()


MODEL_NAME = "llama-3.3-70b-versatile"


def create_llm():
    """Create the Groq LLM."""

    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def generate_answer(
    question: str,
    documents: list[Document],
) -> str:
    """
    Generate a grounded answer using retrieved documents.
    """

    context_parts = []

    for index, document in enumerate(documents, start=1):

        context_parts.append(
            f"""
SOURCE [{index}]

Document:
{document.metadata.get("source", "Unknown")}

Page:
{document.metadata.get("page", "Unknown")}

Content:
{document.page_content}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful AI assistant answering questions
about internal documentation.

Answer the user's question ONLY using the provided
context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the context does not contain enough information,
   say that you do not have enough information.
4. Cite the source supporting each important claim
   using [1], [2], etc.
5. Use the source number corresponding to the context.
6. Keep the answer clear and concise.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    llm = create_llm()

    response = llm.invoke(prompt)

    return response.content