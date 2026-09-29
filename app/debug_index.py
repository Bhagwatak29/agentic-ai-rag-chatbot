from app.retriever import get_vector_store

store = get_vector_store()

# Fetch all indexed chunks from ChromaDB
data = store._collection.get(
    include=["documents", "metadatas"]
)

print(f"Total indexed chunks: {len(data['documents'])}")

print("\nSearching stored chunks for the definition...\n")

for text, metadata in zip(
    data["documents"], data["metadatas"]
):
    normalized = text.lower()

    if (
        "autonomous decision-making and action" in normalized
        or "specific objectives" in normalized
    ):
        print("=" * 60)
        print(f"Page: {metadata.get('page')}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        print(text)
        print("=" * 60)
