import re

from langchain_core.documents import Document
from langchain_groq import ChatGroq


def extract_citations(answer: str) -> list[int]:
    """Extract citation numbers such as [1], [2], [3]."""

    citations = re.findall(r"\[(\d+)\]", answer)

    return sorted(set(int(citation) for citation in citations))


def verify_citation(
    claim: str,
    source: Document,
    llm: ChatGroq,
) -> bool:
    """Check whether a source supports a claim."""

    prompt = f"""
You are a citation verification system.

Determine whether the SOURCE supports the CLAIM.

CLAIM:
{claim}

SOURCE:
{source.page_content}

Answer with exactly one word:

SUPPORTED

or

UNSUPPORTED
"""

    response = llm.invoke(prompt)

    result = response.content.strip().upper()

    return result == "SUPPORTED"