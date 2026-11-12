"""Streamlit chat interface for the document-grounded RAG system."""

import os
from pathlib import Path

import requests
import streamlit as st

from full_pipeline.pipeline import RAGPipeline, index_documents


ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource(show_spinner="Loading the local embedding model...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources used ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata") or {}
            name = metadata.get("source", "Unknown document")
            page = metadata.get("page")
            label = f"[S{index}] {name}"
            if page is not None:
                label += f" - page {page}"
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.caption(
                    f"Cosine similarity: {source.get('score', 0.0):.3f}"
                )
                st.write(source["text"])


def render_assistant_message(message: dict) -> None:
    st.markdown(message["content"])
    if message.get("sources"):
        render_sources(message["sources"])
    if message.get("plain_answer") or message.get("plain_error"):
        with st.expander("Compare: without document retrieval", expanded=False):
            if message.get("plain_answer"):
                st.write(message["plain_answer"])
            elif message.get("plain_error"):
                st.error(message["plain_error"])


def main() -> None:
    st.set_page_config(
        page_title="Grounded | Your document copilot",
        page_icon="G",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        .stApp { background: radial-gradient(ellipse at 12% 0%, #e8f4ff 0, transparent 34%), #f7f9fc; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.035em; }
        p, label, .stCaption { font-family: 'DM Sans', sans-serif; }
        [data-testid="stSidebar"] { background: #101a2b; }
        [data-testid="stSidebar"] * { color: #e8eef8; }
        .hero { padding: 2rem 0 1rem; }
        .eyebrow { color: #3677ed; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
        .hero h1 { font-size: clamp(2.4rem, 5vw, 4rem); line-height: 1.05; margin: .45rem 0; color: #101a2b; }
        .hero p { color: #607087; font-size: 1.08rem; max-width: 720px; }
        .trust { border: 1px solid #dfe7f1; background: rgba(255,255,255,.82); border-radius: 14px; padding: 1rem 1.15rem; color: #43536b; }
        div[data-testid="stChatMessage"] { border: 1px solid #e6ebf2; border-radius: 16px; background: rgba(255,255,255,.88); }
        div.stButton > button { border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    raw_dir = ROOT / "data" / "raw"
    user_docs_dir = ROOT / "data" / "user_docs"
    pdf_count = len(list(raw_dir.glob("*.pdf")))
    if user_docs_dir.exists():
        pdf_count += len(list(user_docs_dir.glob("*.pdf")))
    chunks_path = ROOT / "data" / "processed" / "chunks.jsonl"
    chunk_count = (
        sum(1 for line in chunks_path.open(encoding="utf-8") if line.strip())
        if chunks_path.exists()
        else 0
    )

    with st.sidebar:
        st.markdown("## Grounded")
        st.caption("YOUR KNOWLEDGE. ANSWERS WITH RECEIPTS.")
        st.divider()
        st.markdown("### Knowledge base")
        st.metric("PDF documents", pdf_count)
        st.metric("Indexed passages", chunk_count)
        st.caption("Local ChromaDB + semantic search")
        uploaded_pdfs = st.file_uploader(
            "Add your PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Files are stored locally in data/user_docs and are not uploaded to Groq.",
        )
        if st.button("Save & rebuild local index", use_container_width=True):
            if not uploaded_pdfs:
                st.warning("Choose at least one PDF to add.")
            else:
                user_docs_dir.mkdir(parents=True, exist_ok=True)
                pending_files = []
                upload_error = None
                for uploaded_pdf in uploaded_pdfs:
                    filename = Path(uploaded_pdf.name).name
                    if not filename or Path(filename).suffix.lower() != ".pdf":
                        upload_error = "One of the selected files does not have a PDF filename."
                        break
                    destination = user_docs_dir / filename
                    content = uploaded_pdf.getvalue()
                    if destination.exists() and destination.read_bytes() != content:
                        upload_error = (
                            f"{filename} already exists in data/user_docs. "
                            "Rename the file before adding a different version."
                        )
                        break
                    pending_files.append((destination, content))

                if upload_error:
                    st.error(upload_error)
                else:
                    for destination, content in pending_files:
                        destination.write_bytes(content)
                    try:
                        with st.spinner("Extracting, chunking, and embedding your PDFs..."):
                            indexed_count = index_documents()
                        get_pipeline.clear()
                        st.success(f"Indexed {indexed_count:,} passages.")
                        st.rerun()
                    except (RuntimeError, ValueError, OSError) as error:
                        st.error(str(error))
        st.divider()
        st.markdown("### Answer engine")
        api_key = st.text_input(
            "Groq API key",
            type="password",
            placeholder="gsk_...",
            help="Optional here if GROQ_API_KEY is already set in your .env file.",
        )
        if api_key or os.getenv("GROQ_API_KEY"):
            st.success("API key ready")
        else:
            st.info("Add a Groq key to generate answers.")
        top_k = st.slider("Sources per answer", min_value=2, max_value=8, value=5)
        compare_without_retrieval = st.checkbox(
            "Compare with plain LLM",
            value=False,
            help="Makes one additional Groq request without any document excerpts.",
        )
        st.caption("Your key stays in this local app session.")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">A private workspace for your documents</div>
          <h1>Answers grounded<br>in what you know.</h1>
          <p>Ask a question. Grounded searches your documents, brings back the best passages,
          and gives you a cited answer instead of a confident guess.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="trust">Searching <b>{chunk_count:,}</b> indexed passages across '
        f'<b>{pdf_count}</b> PDFs &nbsp;&middot;&nbsp; Every answer links back to its source.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown("**Try a question**")
    suggestions = [
        "How does retrieval-augmented generation work?",
        "What is the Inverse Cloze Task used for?",
        "How do dense and sparse retrieval differ?",
    ]
    columns = st.columns(len(suggestions))
    for column, suggestion in zip(columns, suggestions):
        if column.button(suggestion, use_container_width=True):
            st.session_state["pending_question"] = suggestion

    st.session_state.setdefault("messages", [])
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])

    question = st.chat_input("Ask something about your documents...")
    pending_question = st.session_state.pop("pending_question", None)
    question = question or pending_question
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Searching your documents and writing a cited answer..."):
            pipeline = get_pipeline()
            result = pipeline.answer(
                question,
                top_k=top_k,
                api_key=api_key,
            )
        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
        if compare_without_retrieval:
            try:
                with st.spinner("Asking the plain LLM without your documents..."):
                    assistant_message["plain_answer"] = (
                        pipeline.answer_without_retrieval(question, api_key=api_key)
                    )
            except (RuntimeError, ValueError, requests.RequestException) as error:
                assistant_message["plain_error"] = str(error)
        st.session_state["messages"].append(assistant_message)
        with st.chat_message("assistant"):
            render_assistant_message(assistant_message)
    except (RuntimeError, ValueError, requests.RequestException) as error:
        st.error(str(error))


if __name__ == "__main__":
    main()