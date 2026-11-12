
"""
Module 04: Vector Database and Storage

Project: RAG_Own_Decision_Making_System

Responsibilities:
1. Initialize persistent ChromaDB.
2. Create and manage collections.
3. Load embeddings from Module 03.
4. Store vectors, text and metadata.
5. Support insert and update operations.
6. Generate query embeddings.
7. Perform semantic similarity searches.
8. Retrieve relevant document chunks.
"""

import json
import logging

from pathlib import Path
from typing import Optional

import chromadb
import numpy as np

from sentence_transformers import SentenceTransformer


# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings.jsonl"
)

DATABASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "vector_db"
    / "chroma"
)

COLLECTION_NAME = "rag_document_collection"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_DIMENSION = 384


# ==========================================================
# 2. LOGGING CONFIGURATION
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# 3. CHROMA VECTOR STORE CLASS
# ==========================================================

class ChromaVectorStore:
    """
    Persistent vector database using ChromaDB.
    """

    def __init__(
        self,
        database_dir: Path = DATABASE_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        model_name: str = MODEL_NAME
    ):

        self.database_dir = Path(database_dir)

        self.collection_name = collection_name

        self.embedding_dimension = embedding_dimension

        self.model_name = model_name

        self.client = None

        self.collection = None

        self.model = None

        logger.info("Chroma Vector Store Initialized")


    # ======================================================
    # 4. INITIALIZE DATABASE
    # ======================================================

    def initialize_database(self) -> None:
        """
        Initialize persistent ChromaDB client.
        """

        self.database_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=str(self.database_dir)
        )

        logger.info(
            f"Database initialized: {self.database_dir}"
        )


    # ======================================================
    # 5. CREATE OR LOAD COLLECTION
    # ======================================================

    def create_collection(self) -> None:
        """
        Create a collection or load an existing one.

        Cosine distance is used for semantic similarity.
        """

        if self.client is None:

            raise RuntimeError(
                "Initialize the database first."
            )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,

            metadata={
                "description": "RAG document embeddings",

                "embedding_model": self.model_name,

                "embedding_dimension": self.embedding_dimension,

                "hnsw:space": "cosine"
            }
        )

        # Verify collection configuration
        collection_metadata = self.collection.metadata or {}

        existing_model = collection_metadata.get(
            "embedding_model"
        )

        if existing_model and existing_model != self.model_name:

            raise ValueError(
                "Embedding model mismatch. "
                "Use the same embedding model used to create "
                "the stored vectors."
            )

        existing_dimension = collection_metadata.get(
            "embedding_dimension"
        )

        if (
            existing_dimension is not None
            and existing_dimension != self.embedding_dimension
        ):

            raise ValueError(
                "Embedding dimension mismatch."
            )

        logger.info(
            f"Collection ready: {self.collection_name}"
        )


    # ======================================================
    # 6. LOAD EMBEDDING MODEL FOR USER QUERIES
    # ======================================================

    def load_embedding_model(self) -> None:
        """
        Load the same model used in Module 03.

        It is needed to convert user queries into vectors.
        """

        self.model = SentenceTransformer(
            self.model_name
        )

        actual_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        if actual_dimension != self.embedding_dimension:

            raise ValueError(
                f"Expected {self.embedding_dimension} dimensions, "
                f"but model generates {actual_dimension}."
            )

        logger.info(
            "Query embedding model loaded successfully."
        )


    # ======================================================
    # 7. READ EMBEDDINGS FROM MODULE 03
    # ======================================================

    @staticmethod
    def load_embeddings(input_file: Path = INPUT_FILE) -> list[dict]:
        """
        Load generated embeddings from JSONL.
        """

        input_file = Path(input_file)

        if not input_file.exists():

            raise FileNotFoundError(
                f"Embeddings file not found: {input_file}"
            )

        records = []

        with input_file.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1
            ):

                if not line.strip():
                    continue

                try:

                    record = json.loads(line)

                    if not record.get("chunk_id"):
                        raise ValueError("Missing chunk_id")

                    if not record.get("text", "").strip():
                        raise ValueError("Missing text")

                    records.append(record)

                except (json.JSONDecodeError, ValueError) as error:

                    logger.warning(
                        f"Invalid record at line {line_number}: {error}"
                    )

        logger.info(
            f"Loaded {len(records)} embedding records."
        )

        return records


    # ======================================================
    # 8. CONVERT METADATA INTO CHROMADB FORMAT
    # ======================================================

    @staticmethod
    def clean_metadata(metadata: dict) -> dict:
        """
        ChromaDB metadata supports scalar values.

        Remove unsupported nested objects, arrays and null values.
        """

        cleaned = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):

                if isinstance(value, float):
                    if not np.isfinite(value):
                        continue

                cleaned[str(key)] = value

            elif isinstance(value, list):

                # ChromaDB supports homogeneous metadata arrays
                if value and all(
                    isinstance(item, str)
                    for item in value
                ):

                    cleaned[str(key)] = value

        return cleaned


    # ======================================================
    # 9. INSERT OR UPDATE VECTORS
    # ======================================================

    def add_embeddings(
        self,
        records: list[dict],
        batch_size: int = 100
    ) -> int:
        """
        Insert or update embeddings using ChromaDB upsert.

        Upsert prevents duplicate IDs when the same dataset
        is processed again.
        """

        if self.collection is None:

            raise RuntimeError(
                "Create the collection first."
            )

        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if not records:
            logger.warning("No embeddings to insert.")
            return 0

        total_inserted = 0

        for start in range(0, len(records), batch_size):

            batch = records[start:start + batch_size]

            ids = []

            documents = []

            embeddings = []

            metadatas = []

            for record in batch:

                vector = np.asarray(
                    record["embedding"],
                    dtype=np.float32
                )

                if vector.shape != (self.embedding_dimension,):

                    raise ValueError(
                        f"Invalid vector dimension for "
                        f"{record['chunk_id']}: {vector.shape}"
                    )

                if not np.isfinite(vector).all():

                    raise ValueError(
                        f"Non-finite vector: {record['chunk_id']}"
                    )

                ids.append(str(record["chunk_id"]))

                documents.append(record["text"])

                embeddings.append(vector.tolist())

                metadata = self.clean_metadata(
                    record.get("metadata", {})
                )

                metadatas.append(metadata)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            total_inserted += len(batch)

            logger.info(
                f"Stored {total_inserted}/{len(records)} vectors."
            )

        logger.info("Vector insertion completed.")

        return total_inserted


    # ======================================================
    # 10. GENERATE QUERY EMBEDDING
    # ======================================================

    def generate_query_embedding(self, query: str) -> list[float]:
        """
        Convert user query into a numerical vector.
        """

        if self.model is None:

            raise RuntimeError(
                "Load the embedding model first."
            )

        if not query or not query.strip():

            raise ValueError("Query cannot be empty.")

        vector = self.model.encode(
            query.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if vector.shape != (self.embedding_dimension,):

            raise ValueError("Query vector dimension mismatch.")

        return vector.astype(np.float32).tolist()


    # ======================================================
    # 11. SEMANTIC SIMILARITY SEARCH
    # ======================================================

    def similarity_search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        """
        Retrieve the most relevant document chunks
        for a given user query.
        """

        if self.collection is None:

            raise RuntimeError(
                "Create the collection first."
            )

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        total_records = self.collection.count()

        if total_records == 0:
            logger.warning("Vector database is empty.")
            return []

        query_vector = self.generate_query_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_vector],

            n_results=min(top_k, total_records),

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved = []

        for index, chunk_id in enumerate(
            results["ids"][0]
        ):

            distance = results["distances"][0][index]

            retrieved.append({

                "chunk_id": chunk_id,

                "text": results["documents"][0][index],

                "metadata": results["metadatas"][0][index],

                "distance": float(distance)

            })

        return retrieved


    # ======================================================
    # 12. DATABASE STATISTICS
    # ======================================================

    def display_statistics(self) -> None:
        """
        Display collection and database information.
        """

        if self.collection is None:

            raise RuntimeError(
                "Create the collection first."
            )

        print("\n" + "=" * 55)

        print("          CHROMADB DATABASE SUMMARY")

        print("=" * 55)

        print(f"Collection Name : {self.collection_name}")

        print(f"Total Vectors   : {self.collection.count()}")

        print(f"Vector Dimension: {self.embedding_dimension}")

        print(f"Distance Metric : Cosine")

        print(f"Database Path   : {self.database_dir}")

        print("=" * 55)


# ==========================================================
# 13. MAIN EXECUTION
# ==========================================================

def main():

    # Initialize vector store
    store = ChromaVectorStore()

    # Step 1: Initialize persistent database
    store.initialize_database()

    # Step 2: Create or load collection
    store.create_collection()

    # Step 3: Load existing embeddings from Module 03
    records = store.load_embeddings()

    # Step 4: Insert vectors into ChromaDB
    store.add_embeddings(records)

    # Step 5: Load query embedding model
    store.load_embedding_model()

    # Step 6: Display database statistics
    store.display_statistics()

    # Step 7: Test semantic search
    query = "What is artificial intelligence?"

    results = store.similarity_search(
        query=query,
        top_k=5
    )

    print("\nSEMANTIC SEARCH RESULTS")

    print("-" * 55)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")

        print(f"Chunk ID: {result['chunk_id']}")

        print(f"Distance: {result['distance']:.4f}")

        print(f"Source: {result['metadata'].get('source', 'Unknown')}")

        print(f"Page: {result['metadata'].get('page', 'Unknown')}")

        print(f"Text: {result['text'][:300]}")

        print("-" * 55)


# Execute module
if __name__ == "__main__":

    main()
