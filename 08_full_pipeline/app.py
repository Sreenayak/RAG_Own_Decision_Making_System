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
STEP_LABELS = [
    "01 - Document Loading",
    "02 - Chunking",
    "03 - Embeddings",
    "04 - Vector Database",
    "05 - Retrieval",
    "06 - Prompting",
    "07 - LLM Integration",
    "08 - Full Pipeline",
    "09 - Evaluation",
    "10 - Advanced Techniques",
]

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
    total = 0
    indexed_names = []

    for file_obj in files:
        file_path = getattr(file_obj, "name", file_obj)  # str path (Gradio 5+) or file object
        try:
            n = p.index_file(file_path)
            total += n
            indexed_names.append(Path(file_path).name)
        except Exception as e:
            return f"❌ Error indexing {Path(file_path).name}: {e}", status_text

    sources = p.get_indexed_sources()
    sources_str = "\n".join(f"• {s}" for s in sources) if sources else "None"
    status = f"✅ Indexed {total} chunks from {len(indexed_names)} file(s).\n\nIndexed documents:\n{sources_str}"
    return status, sources_str


def answer_question(
    question: str,
    history: list,
    top_k: int,
    show_sources: bool,
) -> tuple[list, str]:
    """
    Called when the user submits a question.
    Returns updated chat history and sources text.
    """
    if not question.strip():
        return history, ""

    p = get_pipeline()

    # Check if anything is indexed
    sources_list = p.get_indexed_sources()
    if not sources_list:
        history = history + _turn(question, "⚠️ No documents indexed yet. Please upload a PDF first.")
        return history, ""

    try:
        response = p.query(question, top_k=int(top_k))
    except Exception as e:
        history = history + _turn(question, f"❌ Error: {e}")
        return history, ""

    # Build sources text
    sources_md = ""
    if show_sources and response.sources:
        lines = [f"**Sources ({response.num_chunks_retrieved} chunks retrieved | {response.latency_s:.1f}s):**\n"]
        for i, s in enumerate(response.sources, 1):
            text_preview = s.get("text", "")[:120].replace("\n", " ")
            lines.append(f"**[{i}]** `{s.get('source', '?')}` — {text_preview}...")
        sources_md = "\n".join(lines)

    history = history + _turn(question, response.answer)
    return history, sources_md


def stream_answer(question: str, history: list, top_k: int):
    """Stream tokens for a smoother demo experience."""
    if not question.strip():
        yield history, ""
        return

    try:
        p = get_pipeline()
    except Exception as exc:
        history = history + _turn(question, f"Pipeline setup failed: {exc}")
        yield history, ""
        return

    if not p.get_indexed_sources():
        history = history + _turn(question, "⚠️ No documents indexed yet. Please upload a PDF first.")
        yield history, ""
        return

    history = history + _turn(question, "")
    collected = ""

    try:
        for token in p.stream_query(question, top_k=int(top_k)):
            collected += token
            history[-1] = {"role": "assistant", "content": collected}
            yield history, ""
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"❌ Error: {e}"}
        yield history, ""


def reset_index() -> tuple[str, str, list]:
    """Clear the entire index."""
    p = get_pipeline()
    p.reset_index()
    return "🗑️ Index cleared. Upload new documents to start fresh.", "", []


def show_step(step: str) -> list:
    return [gr.update(visible=(label == step)) for label in STEP_LABELS]


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
#step-content {
    min-width: 0;
}
.step-panel {
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
            gr.Markdown("### Pipeline Steps")
            step_nav = gr.Radio(
                choices=STEP_LABELS,
                value=STEP_LABELS[0],
                label="Navigate",
                elem_id="step-navigation",
            )
            gr.Markdown("Work through each stage from document loading to evaluation.")

        with gr.Column(scale=4, elem_id="step-content"):
            with gr.Column(visible=True, elem_classes=["step-panel"]) as documents_page:
                gr.Markdown("""
## 01 - Document Loading
Upload PDF files to extract their text and add them to the local knowledge base.
                """)
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="Upload PDF files",
                            file_count="multiple",
                            file_types=[".pdf"],
                        )
                        upload_btn = gr.Button("Index Documents", variant="primary")
                        reset_btn = gr.Button("Reset Index", variant="stop")
                    with gr.Column(scale=3):
                        index_status = gr.Textbox(
                            label="Indexing Status",
                            lines=8,
                            interactive=False,
                            placeholder="Upload files and click Index...",
                        )
                        sources_display = gr.Textbox(
                            label="Indexed Sources",
                            lines=4,
                            interactive=False,
                            placeholder="No documents indexed yet.",
                        )

            with gr.Column(visible=False, elem_classes=["step-panel"]) as chunking_page:
                gr.Markdown("""
## 02 - Chunking
Documents are split into overlapping text chunks before embedding. Adjust the active strategy in `config.yaml`.
                """)
                gr.JSON(label="Active chunking settings", value=APP_CONFIG.get("chunking", {}))

            with gr.Column(visible=False, elem_classes=["step-panel"]) as embeddings_page:
                gr.Markdown("""
## 03 - Embeddings
The embedding model turns each chunk into a vector so semantically similar text can be retrieved.
                """)
                gr.JSON(label="Embedding settings", value=APP_CONFIG.get("embeddings", {}))

            with gr.Column(visible=False, elem_classes=["step-panel"]) as vector_db_page:
                gr.Markdown("""
## 04 - Vector Database
Vectors and their source text are persisted in the configured vector database.
                """)
                gr.JSON(label="Vector database settings", value=APP_CONFIG.get("vector_db", {}))

            with gr.Column(visible=False, elem_classes=["step-panel"]) as retrieval_page:
                gr.Markdown("""
## 05 - Retrieval
Ask a question to retrieve relevant chunks. The configured top-k and retrieval strategy are shown below.
                """)
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="Conversation", height=450)
                        with gr.Row():
                            question_box = gr.Textbox(
                                label="Your Question",
                                placeholder="Ask about your indexed documents...",
                                scale=5,
                            )
                            ask_btn = gr.Button("Ask", variant="primary", scale=1)
                    with gr.Column(scale=2):
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=APP_CONFIG.get("retrieval", {}).get("top_k", 5),
                            step=1,
                            label="Top-K chunks",
                            info="Number of relevant chunks to retrieve",
                        )
                        clear_btn = gr.Button("Clear Chat")
                        sources_out = gr.Markdown(
                            label="Retrieved Sources",
                            elem_classes=["source-box"],
                        )
                gr.JSON(label="Retrieval settings", value=APP_CONFIG.get("retrieval", {}))

            with gr.Column(visible=False, elem_classes=["step-panel"]) as prompting_page:
                gr.Markdown("""
## 06 - Prompting
The prompt combines the question with retrieved context and instructs the model to ground its answer in that context.
                """)
                gr.JSON(label="Prompt settings", value=APP_CONFIG.get("prompting", {}))

            with gr.Column(visible=False, elem_classes=["step-panel"]) as llm_page:
                gr.Markdown("""
## 07 - LLM Integration
The language model generates the answer after retrieval. Document loading and indexing do not require an LLM API key.
                """)
                gr.JSON(
                    label="LLM settings (no secrets shown)",
                    value=APP_CONFIG.get("llm", {}),
                )
                llm_config = APP_CONFIG.get("llm", {})
                llm_provider = llm_config.get("provider", "ollama")
                llm_provider_config = llm_config.get(llm_provider, {})
                key_env = llm_provider_config.get("api_key_env")
                if key_env and not os.getenv(key_env):
                    gr.Markdown(
                        f"**Setup needed for questions:** set `{key_env}` in `.env`. "
                        "Indexing documents will still work without it."
                    )

            with gr.Column(visible=False, elem_classes=["step-panel"]) as pipeline_page:
                gr.Markdown("""
## 08 - Full Pipeline
The complete query flow is:

1. Embed the question.
2. Retrieve the most relevant chunks.
3. Build a context-grounded prompt.
4. Generate an answer with the configured LLM.

Use the **Document Loading** step to index files and **Retrieval** to ask questions.
                """)

            with gr.Column(visible=False, elem_classes=["step-panel"]) as evaluation_page:
                gr.Markdown("""
## 09 - Evaluation
Evaluate retrieval and answer quality with the RAGAS evaluation module. Configure the evaluation dataset and metrics in `config.yaml`, then run:

```powershell
python .\\09_evaluation\\evaluator.py
```
                """)

            with gr.Column(visible=False, elem_classes=["step-panel"]) as advanced_page:
                gr.Markdown("""
## 10 - Advanced Techniques
Explore hybrid retrieval, reranking, and alternate chunking strategies by changing the project configuration.

This application is the interactive companion to **RAG From Scratch**. The project includes document loading, chunking, embeddings, vector storage, retrieval, prompting, LLM integration, evaluation, and advanced RAG modules.
                """)
                gr.JSON(label="Advanced retrieval settings", value=APP_CONFIG.get("retrieval", {}))

    step_pages = [
        documents_page,
        chunking_page,
        embeddings_page,
        vector_db_page,
        retrieval_page,
        prompting_page,
        llm_page,
        pipeline_page,
        evaluation_page,
        advanced_page,
    ]

    # ── Event wiring ────────────────────────────────────────────────────────────
    step_nav.change(
        fn=show_step,
        inputs=step_nav,
        outputs=step_pages,
    )

    upload_btn.click(
        fn=upload_and_index,
        inputs=[file_input, index_status],
        outputs=[index_status, sources_display],
    )

    reset_btn.click(
        fn=reset_index,
        outputs=[index_status, sources_display, chatbot],
    )

    ask_btn.click(
        fn=stream_answer,
        inputs=[question_box, chatbot, top_k_slider],
        outputs=[chatbot, sources_out],
    )

    question_box.submit(
        fn=stream_answer,
        inputs=[question_box, chatbot, top_k_slider],
        outputs=[chatbot, sources_out],
    )

    clear_btn.click(fn=lambda: [], outputs=[chatbot])


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
