
"""
Module 01: Document Loading
Project: RAG_Own_Decision_Making

Responsibilities:
1. Discover PDF files in data/raw/
2. Extract text page by page
3. Extract document metadata
4. Handle corrupted and empty PDFs
5. Support pypdf and optional Docling
6. Save extracted records into data/processed/documents.jsonl
"""

import json
import logging

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from pypdf import PdfReader


# ==========================================================
# 1. PROJECT PATH CONFIGURATION
# ==========================================================

# Get the root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input directory containing PDF documents
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Output directory for extracted documents
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Output JSONL file
OUTPUT_FILE = PROCESSED_DATA_DIR / "documents.jsonl"


# ==========================================================
# 2. LOGGING CONFIGURATION
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# 3. DOCUMENT LOADER CLASS
# ==========================================================

class DocumentLoader:
    """
    Loads PDF documents and extracts their text and metadata.
    """

    def __init__(
        self,
        input_dir: Path = RAW_DATA_DIR,
        output_file: Path = OUTPUT_FILE,
        extraction_method: str = "pypdf"
    ):

        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)

        self.extraction_method = extraction_method.lower()

        if self.extraction_method not in {"pypdf", "docling"}:
            raise ValueError(
                "Extraction method must be 'pypdf' or 'docling'."
            )

        self.documents = []

        logger.info("Document Loader Initialized")
        logger.info(f"Input Directory: {self.input_dir}")


    # ======================================================
    # 4. FIND ALL PDF DOCUMENTS
    # ======================================================

    def discover_documents(self) -> list[Path]:
        """
        Discover all PDF files inside the raw data directory.
        """

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.input_dir}"
            )

        pdf_files = sorted(
            path for path in self.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )

        if not pdf_files:
            logger.warning("No PDF documents found.")

        else:
            logger.info(f"Found {len(pdf_files)} PDF documents.")

        return pdf_files


    # ======================================================
    # 5. EXTRACT METADATA
    # ======================================================

    def extract_metadata(
        self,
        pdf_path: Path,
        reader: PdfReader,
        page_number: Optional[int] = None
    ) -> dict:
        """
        Extract document information and source details.
        """

        pdf_metadata = reader.metadata

        metadata = {
            "source": pdf_path.name,
            "file_path": str(pdf_path.relative_to(PROJECT_ROOT)),
            "file_type": "pdf",
            "title": (
                str(pdf_metadata.title)
                if pdf_metadata and pdf_metadata.title
                else pdf_path.stem
            ),
            "author": (
                str(pdf_metadata.author)
                if pdf_metadata and pdf_metadata.author
                else "Unknown"
            ),
            "page": page_number,
            "total_pages": len(reader.pages),
            "extraction_method": self.extraction_method,
            "loaded_at": datetime.now(timezone.utc).isoformat()
        }

        return metadata


    # ======================================================
    # 6. LOAD PDF USING PYPDF
    # ======================================================

    def load_with_pypdf(self, pdf_path: Path) -> list[dict]:
        """
        Extract text page by page using pypdf.
        """

        logger.info(f"Loading: {pdf_path.name}")

        extracted_documents = []

        try:

            reader = PdfReader(str(pdf_path), strict=False)

            if reader.is_encrypted:
                try:
                    result = reader.decrypt("")
                    if not result:
                        logger.error(
                            f"Password-protected PDF skipped: {pdf_path.name}"
                        )
                        return []
                except Exception as error:
                    logger.error(
                        f"Unable to decrypt {pdf_path.name}: {error}"
                    )
                    return []

            total_pages = len(reader.pages)

            for page_index, page in enumerate(reader.pages):

                try:

                    # Extract text from the current page
                    text = page.extract_text() or ""

                    text = text.strip()

                    # Ignore pages without extractable text
                    if not text:
                        logger.warning(
                            f"Empty page {page_index + 1}: {pdf_path.name}"
                        )
                        continue

                    # Extract page metadata
                    metadata = self.extract_metadata(
                        pdf_path,
                        reader,
                        page_number=page_index + 1
                    )

                    # Store extracted content
                    document = {
                        "text": text,
                        "metadata": metadata
                    }

                    extracted_documents.append(document)

                except Exception as error:

                    logger.error(
                        f"Page {page_index + 1} failed "
                        f"in {pdf_path.name}: {error}"
                    )

            logger.info(
                f"Extracted {len(extracted_documents)} "
                f"pages from {pdf_path.name} "
                f"(total pages: {total_pages})"
            )

        except Exception as error:

            logger.error(
                f"Failed to load {pdf_path.name}: {error}"
            )

        return extracted_documents


    # ======================================================
    # 7. LOAD PDF USING DOCLING (OPTIONAL)
    # ======================================================

    def load_with_docling(self, pdf_path: Path) -> list[dict]:
        """
        Extract structured document text using IBM Docling.

        Note:
        Docling returns the converted document as Markdown.
        This implementation stores it as one document record.
        """

        try:

            from docling.document_converter import DocumentConverter

            logger.info(f"Loading with Docling: {pdf_path.name}")

            converter = DocumentConverter()

            result = converter.convert(str(pdf_path))

            markdown_text = result.document.export_to_markdown()

            if not markdown_text.strip():
                logger.warning(
                    f"No text extracted from {pdf_path.name}"
                )
                return []

            document = {
                "text": markdown_text,
                "metadata": {
                    "source": pdf_path.name,
                    "file_path": str(pdf_path.relative_to(PROJECT_ROOT)),
                    "file_type": "pdf",
                    "title": pdf_path.stem,
                    "author": "Unknown",
                    "page": None,
                    "total_pages": None,
                    "extraction_method": "docling",
                    "loaded_at": datetime.now(timezone.utc).isoformat()
                }
            }

            logger.info(
                f"Successfully extracted {pdf_path.name} using Docling"
            )

            return [document]

        except ImportError:

            logger.error(
                "Docling is not installed. Run: pip install docling"
            )
            return []

        except Exception as error:

            logger.error(
                f"Docling extraction failed for {pdf_path.name}: {error}"
            )
            return []


    # ======================================================
    # 8. LOAD ALL DOCUMENTS
    # ======================================================

    def load_all_documents(self) -> list[dict]:
        """
        Load every PDF document from the input directory.
        """

        pdf_files = self.discover_documents()

        self.documents = []

        for pdf_path in pdf_files:

            if self.extraction_method == "pypdf":

                extracted = self.load_with_pypdf(pdf_path)

            else:

                extracted = self.load_with_docling(pdf_path)

            self.documents.extend(extracted)

        logger.info(
            f"Total extracted document records: {len(self.documents)}"
        )

        return self.documents


    # ======================================================
    # 9. SAVE EXTRACTED DOCUMENTS
    # ======================================================

    def save_documents(self) -> None:
        """
        Save extracted text and metadata in JSON Lines format.
        """

        if not self.documents:
            logger.warning("No extracted documents to save.")
            return

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.output_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            for document in self.documents:

                file.write(
                    json.dumps(
                        document,
                        ensure_ascii=False
                    ) + "\n"
                )

        logger.info(
            f"Documents saved successfully: {self.output_file}"
        )


    # ======================================================
    # 10. DISPLAY PROCESSING SUMMARY
    # ======================================================

    def display_summary(self) -> None:
        """
        Display document extraction statistics.
        """

        total_records = len(self.documents)

        unique_sources = {
            doc["metadata"]["source"]
            for doc in self.documents
        }

        total_characters = sum(
            len(doc["text"])
            for doc in self.documents
        )

        print("\n" + "=" * 55)
        print("          DOCUMENT LOADING SUMMARY")
        print("=" * 55)

        print(f"PDF Files Processed : {len(unique_sources)}")
        print(f"Extracted Records   : {total_records}")
        print(f"Total Characters    : {total_characters:,}")
        print(f"Output File         : {self.output_file}")

        print("=" * 55)


# ==========================================================
# 11. MAIN EXECUTION
# ==========================================================

def main():

    # Initialize the document loader
    loader = DocumentLoader(
        extraction_method="pypdf"
    )

    # Discover and extract all PDF documents
    loader.load_all_documents()

    # Save extracted text and metadata
    loader.save_documents()

    # Display processing statistics
    loader.display_summary()


# Execute the program
if __name__ == "__main__":
    main()
