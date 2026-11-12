import unittest
from unittest.mock import Mock

from full_pipeline.pipeline import RAGPipeline


class RAGPipelineTests(unittest.TestCase):
    def test_injects_retrieved_passages_and_returns_citations(self):
        source = {
            "text": "Inverse Cloze Task predicts masked text.",
            "metadata": {"source": "retrieval_augmented_generation.pdf", "page": 3},
            "score": 0.52,
        }
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.retriever = Mock()
        pipeline.retriever.retrieve.return_value = [source]
        pipeline.llm = Mock()
        pipeline.llm.generate.return_value = "Inverse Cloze Task (ICT). [S1]"

        result = pipeline.answer("What is the technique?", top_k=2, api_key="key")

        self.assertEqual(result["answer"], "Inverse Cloze Task (ICT). [S1]")
        self.assertEqual(result["sources"], [source])
        messages = pipeline.llm.generate.call_args.args[0]
        self.assertIn("Inverse Cloze Task predicts masked text.", messages[1]["content"])
        self.assertIn("What is the technique?", messages[1]["content"])

    def test_does_not_call_llm_when_index_has_no_results(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.retriever = Mock()
        pipeline.retriever.retrieve.return_value = []
        pipeline.llm = Mock()

        with self.assertRaisesRegex(RuntimeError, "index is empty"):
            pipeline.answer("What is the technique?")
        pipeline.llm.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
