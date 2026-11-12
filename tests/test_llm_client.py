import os
import unittest
from unittest.mock import Mock, patch

from llm_integration.llm_client import GroqLLMClient


class GroqLLMClientTests(unittest.TestCase):
    def test_sends_messages_and_returns_answer(self):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "The answer is ICT. [S1]"}}]
        }
        client = GroqLLMClient(model="openai/gpt-oss-120b")
        messages = [{"role": "user", "content": "What is ICT?"}]

        with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch(
            "llm_integration.llm_client.requests.post",
            return_value=response,
        ) as post:
            answer = client.generate(messages, api_key="test-key")

        self.assertEqual(answer, "The answer is ICT. [S1]")
        self.assertEqual(post.call_args.kwargs["json"]["messages"], messages)
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-key",
        )
        response.raise_for_status.assert_called_once_with()

    def test_requires_an_api_key(self):
        client = GroqLLMClient()
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "Groq API key is not configured"):
                client.generate([{"role": "user", "content": "test"}])


if __name__ == "__main__":
    unittest.main()
