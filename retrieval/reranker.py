
from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        print("Loading Cross-Encoder Reranker...")

        self.model = CrossEncoder(RERANKER_MODEL)

        print("Reranker initialized successfully.")

    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []

        pairs = [
            [query, document["text"]]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        reranked_documents = []

        for document, score in zip(documents, scores):
            updated_document = document.copy()

            updated_document["rerank_score"] = float(score)

            reranked_documents.append(updated_document)

        reranked_documents.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked_documents[:top_k]


if __name__ == "__main__":
    sample_documents = [
        {
            "text": "Artificial intelligence allows machines to perform tasks that normally require human intelligence."
        },
        {
            "text": "Python is a programming language used for software development."
        },
        {
            "text": "Machine learning is a branch of artificial intelligence."
        }
    ]

    reranker = Reranker()

    question = "What is artificial intelligence?"

    results = reranker.rerank(
        question,
        sample_documents,
        top_k=3
    )

    print("\nReranked Results:")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Score:", round(result["rerank_score"], 4))
        print("Text:", result["text"])
