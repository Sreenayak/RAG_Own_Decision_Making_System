import unittest
from unittest.mock import Mock

from full_pipeline.pipeline import RAGPipeline


class RAGPipelineTests(unittest.TestCase):
    def test_uses_chat_history_for_retrieval_and_grounded_answer(self):
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

        history = [
            {"role": "user", "content": "What is the Inverse Cloze Task?"},
            {"role": "assistant", "content": "It is a retrieval pre-training task."},
        ]
        result = pipeline.answer(
            "How does it help?",
            top_k=2,
            api_key="key",
            history=history,
        )

        self.assertEqual(result["answer"], "Inverse Cloze Task (ICT). [S1]")
        self.assertEqual(result["sources"], [source])
        pipeline.retriever.retrieve.assert_called_once_with(
            "What is the Inverse Cloze Task? How does it help?",
            top_k=2,
        )
        messages = pipeline.llm.generate.call_args.args[0]
        self.assertEqual(messages[1], history[0])
        self.assertEqual(messages[2], history[1])
        self.assertIn("Inverse Cloze Task predicts masked text.", messages[3]["content"])
        self.assertIn("How does it help?", messages[3]["content"])

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
