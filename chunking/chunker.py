
"""
Module 02: Text Preprocessing and Chunking

Project: RAG_Own_Decision_Making_System

Responsibilities:
1. Read extracted documents from JSONL.
2. Clean and normalize text.
3. Perform token-based chunking.
4. Support overlapping chunks.
5. Support sentence-aware chunking.
6. Preserve document metadata.
7. Save processed chunks into JSONL.
"""

import json
import re
import logging

from pathlib import Path
from typing import Optional

import tiktoken


# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"

OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


# ==========================================================
# 2. LOGGING CONFIGURATION
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# 3. TEXT CHUNKER CLASS
# ==========================================================

class TextChunker:
    """
    Preprocesses extracted documents and creates text chunks.
    """

    def __init__(
        self,
        input_file: Path = INPUT_FILE,
        output_file: Path = OUTPUT_FILE,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        strategy: str = "token"
    ):

        self.input_file = Path(input_file)
        self.output_file = Path(output_file)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.strategy = strategy.lower()

        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "Chunk overlap must be >= 0 and smaller than chunk size."
            )

        if self.strategy not in {"token", "sentence"}:
            raise ValueError(
                "Strategy must be 'token' or 'sentence'."
            )

        # Initialize tokenizer
        self.encoding = tiktoken.get_encoding("cl100k_base")

        self.documents = []
        self.chunks = []

        logger.info("Text Chunker Initialized")
        logger.info(f"Chunk Size: {self.chunk_size}")
        logger.info(f"Chunk Overlap: {self.chunk_overlap}")
        logger.info(f"Strategy: {self.strategy}")


    # ======================================================
    # 4. READ EXTRACTED DOCUMENTS
    # ======================================================

    def load_documents(self) -> list[dict]:
        """
        Read document records from the JSONL output of Module 01.
        """

        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Document file not found: {self.input_file}"
            )

        self.documents = []

        with self.input_file.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                try:
                    document = json.loads(line)

                    if "text" not in document:
                        logger.warning(
                            f"Missing text at line {line_number}"
                        )
                        continue

                    if not isinstance(document["text"], str):
                        logger.warning(
                            f"Invalid text at line {line_number}"
                        )
                        continue

                    if not document["text"].strip():
                        continue

                    self.documents.append(document)

                except json.JSONDecodeError as error:

                    logger.warning(
                        f"Invalid JSON at line {line_number}: {error}"
                    )

        logger.info(
            f"Loaded {len(self.documents)} document records."
        )

        return self.documents


    # ======================================================
    # 5. CLEAN AND NORMALIZE TEXT
    # ======================================================

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Remove unnecessary whitespace and normalize text.
        """

        # Remove null characters
        text = text.replace("\x00", " ")

        # Fix words split by PDF line wrapping
        text = re.sub(
            r"(?<=\w)-\s*\n\s*(?=\w)",
            "",
            text
        )

        # Replace line breaks with spaces
        text = re.sub(r"[\r\n\t]+", " ", text)

        # Remove repeated spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    # ======================================================
    # 6. TOKENIZE TEXT
    # ======================================================

    def tokenize(self, text: str) -> list[int]:
        """
        Convert text into token IDs.
        """

        return self.encoding.encode(text)


    # ======================================================
    # 7. TOKEN-BASED CHUNKING
    # ======================================================

    def token_chunking(self, text: str) -> list[dict]:
        """
        Split text into fixed-size token chunks
        with configurable overlap.
        """

        tokens = self.tokenize(text)

        chunks = []

        step = self.chunk_size - self.chunk_overlap

        for start in range(0, len(tokens), step):

            end = min(
                start + self.chunk_size,
                len(tokens)
            )

            chunk_tokens = tokens[start:end]

            chunk_text = self.encoding.decode(chunk_tokens)

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text.strip(),
                    "token_count": len(chunk_tokens),
                    "start_token": start,
                    "end_token": end
                })

            if end == len(tokens):
                break

        return chunks


    # ======================================================
    # 8. SENTENCE-AWARE CHUNKING
    # ======================================================

    def sentence_chunking(self, text: str) -> list[dict]:
        """
        Group complete sentences into chunks while
        respecting the approximate token limit.

        Oversized individual sentences are split into
        smaller token chunks.
        """

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        chunks = []
        current_tokens = []

        def save_chunk(token_ids):

            if token_ids:

                chunks.append({
                    "text": self.encoding.decode(
                        token_ids
                    ).strip(),

                    "token_count": len(token_ids)
                })

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_tokens = self.tokenize(sentence)

            # Handle an individual oversized sentence
            if len(sentence_tokens) > self.chunk_size:

                if current_tokens:
                    save_chunk(current_tokens)
                    current_tokens = []

                oversized_chunks = self.token_chunking(sentence)

                chunks.extend(oversized_chunks)

                continue

            # Check whether the next sentence exceeds chunk size
            if (
                current_tokens
                and len(current_tokens) + len(sentence_tokens)
                > self.chunk_size
            ):

                previous_tokens = current_tokens

                save_chunk(previous_tokens)

                # Preserve overlap from previous chunk
                overlap_tokens = previous_tokens[
                    -self.chunk_overlap:
                ] if self.chunk_overlap else []

                current_tokens = overlap_tokens.copy()

            current_tokens.extend(sentence_tokens)

        # Save remaining tokens
        save_chunk(current_tokens)

        return chunks


    # ======================================================
    # 9. PROCESS ALL DOCUMENTS
    # ======================================================

    def process_documents(self) -> list[dict]:
        """
        Clean documents and create chunks with metadata.
        """

        self.chunks = []

        for document_index, document in enumerate(
            self.documents,
            start=1
        ):

            text = self.clean_text(document["text"])

            if not text:
                continue

            # Apply selected chunking technique
            if self.strategy == "token":

                document_chunks = self.token_chunking(text)

            else:

                document_chunks = self.sentence_chunking(text)

            original_metadata = document.get(
                "metadata",
                {}
            )

            for chunk_index, chunk in enumerate(
                document_chunks,
                start=1
            ):

                chunk_record = {

                    "chunk_id": (
                        f"doc_{document_index}_"
                        f"chunk_{chunk_index}"
                    ),

                    "text": chunk["text"],

                    "metadata": {
                        **original_metadata,

                        "chunk_index": chunk_index,

                        "token_count": chunk["token_count"],

                        "chunking_strategy": self.strategy
                    }
                }

                self.chunks.append(chunk_record)

        logger.info(
            f"Generated {len(self.chunks)} chunks."
        )

        return self.chunks


    # ======================================================
    # 10. SAVE PROCESSED CHUNKS
    # ======================================================

    def save_chunks(self) -> None:
        """
        Save all generated chunks in JSONL format.
        """

        if not self.chunks:
            logger.warning("No chunks available to save.")
            return

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.output_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            for chunk in self.chunks:

                file.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    ) + "\n"
                )

        logger.info(
            f"Chunks saved successfully: {self.output_file}"
        )


    # ======================================================
    # 11. DISPLAY CHUNKING SUMMARY
    # ======================================================

    def display_summary(self) -> None:
        """
        Display preprocessing and chunking statistics.
        """

        total_chunks = len(self.chunks)

        total_tokens = sum(
            chunk["metadata"]["token_count"]
            for chunk in self.chunks
        )

        average_tokens = (
            total_tokens / total_chunks
            if total_chunks else 0
        )

        unique_sources = {
            chunk["metadata"].get("source", "Unknown")
            for chunk in self.chunks
        }

        print("\n" + "=" * 55)
        print("          TEXT CHUNKING SUMMARY")
        print("=" * 55)

        print(f"Input Documents    : {len(self.documents)}")
        print(f"Source PDF Files   : {len(unique_sources)}")
        print(f"Total Chunks       : {total_chunks}")
        print(f"Total Tokens       : {total_tokens:,}")
        print(f"Average Tokens     : {average_tokens:.2f}")
        print(f"Chunk Size         : {self.chunk_size}")
        print(f"Chunk Overlap      : {self.chunk_overlap}")
        print(f"Strategy           : {self.strategy}")
        print(f"Output File        : {self.output_file}")

        print("=" * 55)


# ==========================================================
# 12. MAIN EXECUTION
# ==========================================================

def main():

    # Initialize the chunker
    chunker = TextChunker(
        chunk_size=500,
        chunk_overlap=100,
        strategy="token"
    )

    # Load extracted documents from Module 01
    chunker.load_documents()

    # Clean and generate chunks
    chunker.process_documents()

    # Save chunks
    chunker.save_chunks()

    # Display processing statistics
    chunker.display_summary()


# Execute the module
if __name__ == "__main__":
    main()
