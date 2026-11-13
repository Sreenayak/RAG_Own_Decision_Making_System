"""Retrieval, grounded answer generation, and document indexing."""

import json
from pathlib import Path

from 02_chunking.chunker import TextChunker
from 01_document_loading.loader import DocumentLoader
from 03_embeddings.embedder import EmbeddingGenerator
from 07_llm_integration.llm_client import GroqLLMClient
from 06_prompting.prompt_builder import build_messages
from 05_retrieval.dense_retriever import DenseRetriever
from vector_db.chroma.chroma_store import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


class RAGPipeline:
    """Answer questions with passages retrieved from the local Chroma index."""

    def __init__(self):
        self.retriever = DenseRetriever()
        self.llm = GroqLLMClient()

    @property
    def indexed_chunks(self) -> int:
        return self.retriever.collection.count()

    def answer(
        self,
        question: str,
        top_k: int = 5,
        api_key: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> dict:
        if not question or not question.strip():
            raise ValueError("Enter a question before submitting.")
        if top_k <= 0:
            raise ValueError("The number of retrieved sources must be positive.")
        history = history or []
        if not isinstance(history, list):
            raise ValueError("Conversation history must be a list.")
        for message in history:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError("Conversation history contains an invalid message.")

        previous_question = next(
            (
                message["content"].strip()
                for message in reversed(history)
                if message["role"] == "user" and message["content"].strip()
            ),
            "",
        )
        retrieval_query = f"{previous_question} {question.strip()}".strip()
        sources = self.retriever.retrieve(retrieval_query, top_k=top_k)
        if not sources:
            raise RuntimeError(
                "The Chroma index is empty. Run the document indexing command first."
            )

        messages = build_messages(question.strip(), sources, history=history)
        answer = self.llm.generate(messages, api_key=api_key)
        return {"answer": answer, "sources": sources}

    def answer_without_retrieval(self, question: str, api_key: str = "") -> str:
        if not question or not question.strip():
            raise ValueError("Enter a question before submitting.")
        return self.llm.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer using only your pretrained knowledge. You do not "
                        "have access to the user's files or document index. If you "
                        "do not know the answer, say so; do not imply that you "
                        "searched any documents."
                    ),
                },
                {"role": "user", "content": question.strip()},
            ],
            api_key=api_key,
        )


def index_documents() -> int:
    """Extract the PDFs, chunk and embed them, then update the Chroma index."""
    user_documents = PROJECT_ROOT / "data" / "user_docs"
    user_documents.mkdir(parents=True, exist_ok=True)
    documents = []
    for input_dir in (PROJECT_ROOT / "data" / "raw", user_documents):
        documents.extend(
            DocumentLoader(input_dir=input_dir).load_all_documents()
        )
    if not documents:
        raise RuntimeError("No readable PDF documents were found in data/raw.")

    chunker = TextChunker()
    chunker.documents = documents
    chunks = chunker.process_documents()
    if not chunks:
        raise RuntimeError("No text chunks were created from the PDF documents.")

    embedder = EmbeddingGenerator()
    embedder.documents = chunks
    embedder.load_model()
    records = embedder.generate_embeddings()
    if not records:
        raise RuntimeError("No embeddings were generated.")

    store = ChromaVectorStore()
    store.initialize_database()
    store.create_collection()

    expected_ids = {str(record["chunk_id"]) for record in records}
    existing_ids = store.collection.get(include=[])["ids"]
    stale_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in expected_ids]
    if stale_ids:
        store.collection.delete(ids=stale_ids)

    inserted = store.add_embeddings(records)
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return inserted


if __name__ == "__main__":
    print(f"Indexed {index_documents()} document chunks.")
