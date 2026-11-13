
from 05_retrieval.dense_retriever import DenseRetriever
from 05_retrieval.sparse_retriever import SparseRetriever


class HybridRetriever:
    def __init__(self):
        print("Initializing Hybrid Retriever...")

        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()

        print("Hybrid Retriever initialized successfully.")

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        rrf_k=60
    ):
        fused_results = {}

        result_lists = [
            dense_results,
            sparse_results
        ]

        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):

                metadata = result.get("metadata", {})

                chunk_id = (
                    metadata.get("chunk_id")
                    or metadata.get("id")
                    or result["text"]
                )

                if chunk_id not in fused_results:
                    fused_results[chunk_id] = {
                        "text": result["text"],
                        "metadata": metadata,
                        "rrf_score": 0.0,
                        "retrieval_methods": []
                    }

                fused_results[chunk_id]["rrf_score"] += (
                    1.0 / (rrf_k + rank)
                )

                method = result["retrieval_method"]

                if method not in fused_results[chunk_id][
                    "retrieval_methods"
                ]:
                    fused_results[chunk_id][
                        "retrieval_methods"
                    ].append(method)

        ranked_results = sorted(
            fused_results.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

        return ranked_results

    def retrieve(self, query, top_k=5, candidate_k=10):
        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=candidate_k
        )

        sparse_results = self.sparse_retriever.retrieve(
            query,
            top_k=candidate_k
        )

        hybrid_results = self.reciprocal_rank_fusion(
            dense_results,
            sparse_results
        )

        return hybrid_results[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever()

    question = "What is artificial intelligence?"

    results = retriever.retrieve(
        question,
        top_k=5,
        candidate_k=10
    )

    print("\nHybrid Retrieval Results:")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("RRF Score:", round(result["rrf_score"], 5))
        print("Methods:", result["retrieval_methods"])
        print("Text:", result["text"][:300])
