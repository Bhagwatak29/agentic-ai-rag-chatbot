
import re
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.ingest import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

# ---------------- CONFIGURATION ----------------

CANDIDATE_K = 15
TOP_K = 4

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------- MODEL LOADING ----------------

@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder model once."""

    return CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
    )


# ---------------- VECTOR DATABASE ----------------

@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Connect to the existing persistent ChromaDB."""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Vector database not found. "
            "Run python -m app.ingest first."
        )

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    try:
        client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception as exc:
        raise RuntimeError(
            "Knowledge collection not found. "
            "Run python -m app.ingest again."
        ) from exc

    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


# ---------------- TEXT PROCESSING ----------------

def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )


def is_definition_question(query: str) -> bool:
    query = query.lower().strip()

    definition_patterns = (
        "what is agentic ai",
        "what are the key components",
        "define ",
        "definition of ",
        "meaning of ",
    )

    return query.startswith(definition_patterns)



def get_subject_terms(question: str) -> list[str]:
    """Extract the main subject from a definition question."""

    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "define",
        "meaning",
        "of",
        "definition",
        "explain",
        "tell",
        "me",
        "about",
    }

    return [
        word
        for word in tokenize(question)
        if word not in stop_words
    ]


# ---------------- RETRIEVAL PIPELINE ----------------

def retrieve_documents(
    question: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve chunks using hybrid search,
    rerank them, and return the best results.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    question = question.strip()

    vector_store = get_vector_store()

    # 1. Semantic search using ChromaDB
    vector_results = (
        vector_store.similarity_search_with_score(
            query=question,
            k=CANDIDATE_K,
        )
    )

    # 2. Get all indexed chunks for BM25
    collection = vector_store._collection

    stored_data = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    all_texts = stored_data["documents"]
    all_metadatas = stored_data["metadatas"]

    if not all_texts:
        return []

    # 3. Build keyword search index
    tokenized_corpus = [
        tokenize(text)
        for text in all_texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenize(question)

    keyword_scores = bm25.get_scores(
        query_tokens
    )

    keyword_indices = sorted(
        range(len(keyword_scores)),
        key=lambda index: keyword_scores[index],
        reverse=True,
    )[:CANDIDATE_K]

    # 4. Merge semantic and keyword candidates
    candidates = {}

    for document, distance in vector_results:
        chunk_id = document.metadata.get(
            "chunk_id"
        )

        key = str(chunk_id)

        candidates[key] = {
            "document": document,
            "vector_distance": float(distance),
            "bm25_score": 0.0,
        }

    for index in keyword_indices:
        metadata = all_metadatas[index] or {}

        chunk_id = metadata.get("chunk_id")
        key = str(chunk_id)

        from langchain_core.documents import Document

        document = Document(
            page_content=all_texts[index],
            metadata=metadata,
        )

        if key in candidates:
            candidates[key]["bm25_score"] = float(
                keyword_scores[index]
            )
        else:
            candidates[key] = {
                "document": document,
                "vector_distance": None,
                "bm25_score": float(
                    keyword_scores[index]
                ),
            }

    candidate_items = list(
        candidates.values()
    )

    if not candidate_items:
        return []

    # 5. Cross-encoder reranking
    reranker = get_reranker()

    pairs = [
        (
            question,
            item["document"].page_content,
        )
        for item in candidate_items
    ]

    rerank_scores = reranker.predict(
        pairs
    )

    ranked_results = []

    for item, rerank_score in zip(
        candidate_items,
        rerank_scores,
    ):
        document = item["document"]

        ranked_results.append(
            {
                "content": document.page_content,
                "source": document.metadata.get(
                    "source",
                    "Unknown",
                ),
                "page": document.metadata.get(
                    "page"
                ),
                "chunk_id": document.metadata.get(
                    "chunk_id"
                ),
                "vector_distance": item[
                    "vector_distance"
                ],
                "bm25_score": item[
                    "bm25_score"
                ],
                "rerank_score": float(
                    rerank_score
                ),
                "final_score": float(
                    rerank_score
                ),
            }
        )

    # 6. Boost exact subject matches for definitions
    if is_definition_question(question):

        subject_terms = get_subject_terms(
            question
        )

        definition_phrases = [
            "refers to",
            "is defined as",
            "is a system",
            "are systems capable",
            "is the ability",
        ]

        for item in ranked_results:
            text = item["content"].lower()

            exact_subject_match = (
                bool(subject_terms)
                and all(
                    term in text
                    for term in subject_terms
                )
            )

            has_definition_language = any(
                phrase in text
                for phrase in definition_phrases
            )

            item["final_score"] = (
                item["rerank_score"]
                + (
                    5.0
                    if exact_subject_match
                    else 0.0
                )
                + (
                    3.0
                    if has_definition_language
                    else 0.0
                )
            )

    # 7. Sort by final score
    ranked_results.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return ranked_results[:top_k]


# ---------------- LOCAL TESTING ----------------

if __name__ == "__main__":

    question = input(
        "Enter a question about the Agentic AI eBook: "
    )

    chunks = retrieve_documents(question)

    print("\nRetrieved Context:\n")

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(f"--- Result {index} ---")
        print(f"Page: {chunk['page']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(
            f"Vector distance: "
            f"{chunk['vector_distance']}"
        )
        print(
            f"BM25 score: "
            f"{chunk['bm25_score']:.4f}"
        )
        print(
            f"Reranker score: "
            f"{chunk['rerank_score']:.4f}"
        )
        print(
            f"Final score: "
            f"{chunk['final_score']:.4f}"
        )
        print(f"Content:\n{chunk['content']}\n")
