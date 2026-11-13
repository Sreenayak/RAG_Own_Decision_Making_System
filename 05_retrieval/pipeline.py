
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker


class RetrievalPipeline:
    def __init__(self):
        print("\nStarting Retrieval Pipeline...")

        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()

        print("Retrieval Pipeline ready.")

    def search(
        self,
        query,
        top_k=5,
        candidate_k=10
    ):
        print("\nSearching for relevant documents...")

        # Stage 1: Hybrid retrieval
        candidates = self.hybrid_retriever.retrieve(
            query,
            top_k=candidate_k,
            candidate_k=candidate_k
        )

        print(
            f"Hybrid retrieval found {len(candidates)} candidates."
        )

        # Stage 2: Reranking
        final_results = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k
        )

        print(
            f"Reranking selected {len(final_results)} results."
        )

        return final_results


if __name__ == "__main__":
    pipeline = RetrievalPipeline()

    question = "What is artificial intelligence?"

    results = pipeline.search(
        query=question,
        top_k=5,
        candidate_k=10
    )

    print("\n========== FINAL RETRIEVAL RESULTS ==========")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")

        print(
            "Rerank Score:",
            round(result["rerank_score"], 4)
        )

        print(
            "RRF Score:",
            round(result["rrf_score"], 5)
        )

        print(
            "Retrieval Methods:",
            result["retrieval_methods"]
        )

        print("Content:")
        print(result["text"][:500])

        print("-" * 60)
