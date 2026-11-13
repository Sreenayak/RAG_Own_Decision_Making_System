
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


class SparseRetriever:
    def __init__(self):
        print("Initializing Sparse Retriever...")

        self.documents = self._load_documents()

        if not self.documents:
            raise ValueError(
                "No chunks found. Run Module 02 first."
            )

        self.tokenized_documents = [
            self._tokenize(doc["text"])
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

        print(
            f"Sparse Retriever initialized with "
            f"{len(self.documents)} chunks."
        )

    def _load_documents(self):
        documents = []

        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {CHUNKS_PATH}"
            )

        with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    documents.append(json.loads(line))

        return documents

    @staticmethod
    def _tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    def retrieve(self, query, top_k=5):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:
            document = self.documents[index]

            results.append({
                "text": document["text"],
                "metadata": document.get("metadata", {}),
                "score": float(scores[index]),
                "retrieval_method": "sparse"
            })

        return results


if __name__ == "__main__":
    retriever = SparseRetriever()

    question = "What is artificial intelligence?"

    results = retriever.retrieve(question, top_k=5)

    print("\nSparse Retrieval Results:")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"][:300])
