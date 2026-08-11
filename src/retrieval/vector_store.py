from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHROMA_PATH = "data/chroma"


def create_embedding_model():
    """Create the Hugging Face embedding model."""

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(
    chunks: list[Document],
) -> Chroma:
    """Create ChromaDB and store document chunks."""

    embedding_model = create_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
        collection_name="rag_documents",
    )

    return vector_store