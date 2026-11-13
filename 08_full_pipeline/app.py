"""
Module 8 — Gradio Web UI
=========================
A browser-based interface for the RAG pipeline.
You can upload your own PDFs and ask questions — no terminal needed.

Run:
    python 08_full_pipeline/app.py

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr  # noqa: E402
from src.config_loader import load_config  # noqa: E402

APP_CONFIG = load_config()
PAGE_LABELS = ["Documents", "Ask Questions", "About"]
DEFAULT_TOP_K = APP_CONFIG.get("retrieval", {}).get("top_k", 5)
_llm_config = APP_CONFIG.get("llm", {})
_llm_provider = _llm_config.get("provider", "ollama")
_api_key_env = _llm_config.get(_llm_provider, {}).get("api_key_env")
LLM_SETUP_NOTE = (
    f"Question answering requires {_api_key_env} in your `.env` file."
    if _api_key_env and not os.getenv(_api_key_env)
    else ""
)

def _imp(mp, a):
    spec = importlib.util.spec_from_file_location(a, Path(_ROOT) / mp)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return getattr(m, a)


RAGPipeline = _imp("08_full_pipeline/pipeline.py", "RAGPipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline singleton (shared across Gradio callbacks)
# ─────────────────────────────────────────────────────────────────────────────

pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Gradio callback functions
# ─────────────────────────────────────────────────────────────────────────────

def _turn(question: str, answer: str) -> list[dict]:
    """One chat turn in Gradio's 'messages' format."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def upload_and_index(files, status_text: str) -> tuple[str, str]:
    """
    Called when the user uploads PDF files.
    Returns (status message, updated sources list).
    """
    if not files:
        return "No files uploaded.", status_text

    try:
        p = get_pipeline()
    except Exception as exc:
        return f"Pipeline setup failed: {exc}", status_text
    indexed_names = []

    for file_obj in files:
        file_path = getattr(file_obj, "name", file_obj)  # str path (Gradio 5+) or file object
        try:
            p.index_file(file_path)
            indexed_names.append(Path(file_path).name)
        except Exception as e:
            return f"Could not add {Path(file_path).name}: {e}", status_text

    sources = p.get_indexed_sources()
    sources_str = "\n".join(f"• {s}" for s in sources) if sources else "None"
    document_word = "document" if len(indexed_names) == 1 else "documents"
    status = f"Added {len(indexed_names)} {document_word} to your knowledge base."
    return status, sources_str


def answer_question(question: str, history: list) -> tuple[list, str]:
    if not question.strip():
        return history, ""

    try:
        p = get_pipeline()
    except Exception as exc:
        return history + _turn(question, f"Pipeline setup failed: {exc}"), ""

    sources_list = p.get_indexed_sources()
    if not sources_list:
        history = history + _turn(
            question,
            "No documents have been added yet. Upload a PDF in the Documents section first.",
        )
        return history, ""

    try:
        response = p.query(question, top_k=DEFAULT_TOP_K)
    except Exception as exc:
        history = history + _turn(question, f"Unable to answer the question: {exc}")
        return history, ""

    sources_md = ""
    if response.sources:
        lines = ["**Sources:**\n"]
        for i, s in enumerate(response.sources, 1):
            lines.append(f"{i}. `{s.get('source', 'Unknown document')}`")
        sources_md = "\n".join(lines)

    history = history + _turn(question, response.answer)
    return history, sources_md


def reset_index() -> tuple[str, str, list, str]:
    """Clear the entire index."""
    p = get_pipeline()
    p.reset_index()
    return "Knowledge base cleared. Upload documents to start again.", "", [], ""


def show_page(page: str) -> list:
    return [gr.update(visible=(label == page)) for label in PAGE_LABELS]


# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
#header {
    text-align: center;
    padding: 1rem 0 1.5rem;
}
#header h1 {
    color: #312e81;
    font-size: 2rem;
    letter-spacing: -0.03em;
    margin: 0;
}
#header p {
    color: #475569;
    margin: 0.45rem 0 0;
}
.source-box {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 0.9rem;
    font-size: 0.9rem;
}
#app-layout {
    align-items: flex-start;
    gap: 1.25rem;
}
#sidebar {
    position: sticky;
    top: 1rem;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 1rem;
}
#sidebar h3 {
    color: #312e81;
    margin-top: 0;
}
#page-content {
    min-width: 0;
}
.page-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
}
@media (max-width: 760px) {
    #sidebar {
        position: static;
    }
}
"""

with gr.Blocks(title="RAG Knowledge Assistant") as demo:

    # ── Header ─────────────────────────────────────────────────────────────────
    gr.HTML("""
        <div id="header">
            <h1>📚 RAG Knowledge Assistant</h1>
            <p>Ask grounded questions about your documents</p>
        </div>
    """)

    with gr.Row(elem_id="app-layout"):
        with gr.Column(scale=1, min_width=220, elem_id="sidebar"):
            gr.Markdown("### Your Workspace")
            page_nav = gr.Radio(
                choices=PAGE_LABELS,
                value=PAGE_LABELS[0],
                label="Menu",
                elem_id="page-navigation",
            )
            gr.Markdown("Upload your files, then ask questions about them.")

        with gr.Column(scale=4, elem_id="page-content"):
            with gr.Column(visible=True, elem_classes=["page-panel"]) as documents_page:
                gr.Markdown("""
## Your Documents
Upload PDF files to make them available for questions.
                """)
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="Choose PDF files",
                            file_count="multiple",
                            file_types=[".pdf"],
                        )
                        upload_btn = gr.Button("Add Documents", variant="primary")
                        clear_documents_btn = gr.Button("Clear All Documents", variant="stop")
                    with gr.Column(scale=3):
                        index_status = gr.Textbox(
                            label="Upload Status",
                            lines=4,
                            interactive=False,
                            placeholder="Choose files and click Add Documents.",
                        )
                        sources_display = gr.Textbox(
                            label="Available Documents",
                            lines=8,
                            interactive=False,
                            placeholder="No documents uploaded yet.",
                        )

            with gr.Column(visible=False, elem_classes=["page-panel"]) as questions_page:
                gr.Markdown("""
## Ask Questions
Ask a question about the documents you have added. Answers are based on the information found in those files.
                """)
                if LLM_SETUP_NOTE:
                    gr.Markdown(
                        f"**Setup needed:** {LLM_SETUP_NOTE} "
                        "You can still upload and prepare documents without it."
                    )
                chatbot = gr.Chatbot(label="Conversation", height=450)
                with gr.Row():
                    question_box = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask something about your documents...",
                        scale=5,
                    )
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)
                clear_chat_btn = gr.Button("Clear Conversation")
                sources_out = gr.Markdown(
                    label="Sources",
                    elem_classes=["source-box"],
                )

            with gr.Column(visible=False, elem_classes=["page-panel"]) as about_page:
                gr.Markdown("""
## About RAG Knowledge Assistant
Use this app to search your PDF documents and get answers grounded in their contents.

### Getting started
1. Open **Documents** and add one or more PDF files.
2. Open **Ask Questions** and enter your question.
3. Check the listed sources to see which documents informed the answer.
                """)
                if LLM_SETUP_NOTE:
                    gr.Markdown(f"To generate answers, {LLM_SETUP_NOTE}")

    pages = [documents_page, questions_page, about_page]

    # ── Event wiring ────────────────────────────────────────────────────────────
    page_nav.change(
        fn=show_page,
        inputs=page_nav,
        outputs=pages,
    )

    upload_btn.click(
        fn=upload_and_index,
        inputs=[file_input, index_status],
        outputs=[index_status, sources_display],
    )

    clear_documents_btn.click(
        fn=reset_index,
        outputs=[index_status, sources_display, chatbot, sources_out],
    )

    ask_btn.click(
        fn=answer_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, sources_out],
    )

    question_box.submit(
        fn=answer_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, sources_out],
    )

    clear_chat_btn.click(fn=lambda: ([], ""), outputs=[chatbot, sources_out])


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css=CSS,
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # set True to get a public URL
        inbrowser=True,
    )
