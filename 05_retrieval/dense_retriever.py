
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHROMA_PATH = PROJECT_ROOT / "data" / "vector_db" / "chroma"
COLLECTION_NAME = "rag_document_collection"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(self):
        print("Initializing Dense Retriever...")

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print("Dense Retriever initialized successfully.")

    def retrieve(self, query, top_k=5):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        total_records = self.collection.count()

        if total_records == 0:
            return []

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_records),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_documents = []

        for i, document in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]

            retrieved_documents.append({
                "text": document,
                "metadata": results["metadatas"][0][i],
                "distance": float(distance),
                "score": 1.0 - float(distance),
                "retrieval_method": "dense"
            })

        return retrieved_documents


if __name__ == "__main__":
    retriever = DenseRetriever()

    question = "What is artificial intelligence?"

    results = retriever.retrieve(question, top_k=5)

    print("\nDense Retrieval Results:")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"][:300])
        print("Metadata:", result["metadata"])
