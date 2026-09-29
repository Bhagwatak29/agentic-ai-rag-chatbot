
from pathlib import Path

import pymupdf  # PyMuPDF
import chromadb

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "Ebook-Agentic-AI.pdf"
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "agentic_ai_knowledge"

# Embedding model runs locally on your computer
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf(pdf_path: Path) -> list[Document]:
    """Extract text from the PDF and preserve page metadata."""

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Place the Agentic AI eBook inside the data folder."
        )

    documents = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )

    if not documents:
        raise ValueError(
            "No readable text found in the PDF. "
            "The document may require OCR."
        )

    print(f"Extracted text from {len(documents)} pages.")

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """Split pages into smaller overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    print(f"Created {len(chunks)} text chunks.")

    return chunks


def create_vector_store(
    chunks: list[Document],
) -> Chroma:
    """Generate embeddings and save chunks in ChromaDB."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Create a persistent local ChromaDB client
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Remove the previous collection during re-ingestion
    existing_collections = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME in existing_collections:
        client.delete_collection(
            name=COLLECTION_NAME
        )

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    ids = [
        f"chunk-{index}"
        for index in range(len(chunks))
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    print(
        f"Stored {len(chunks)} chunks in ChromaDB."
    )

    return vector_store


def main():
    print("Starting PDF ingestion...")

    documents = load_pdf(PDF_PATH)

    chunks = split_documents(documents)

    create_vector_store(chunks)

    print("PDF ingestion completed successfully!")
    print(f"Vector database location: {DB_PATH}")


if __name__ == "__main__":
    main()
