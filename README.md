# Smash AI

Smash AI is a local chat website for asking questions about your knowledge base.
It retrieves relevant passages from the existing Chroma vector database and
processed chunks, then uses Groq to generate an answer from those passages.
The website shows a chat conversation only; it does not show PDF names, file
lists, or document counts.

## Interface features

- Search, reopen, and delete previous conversations; history stays in this
  browser's local storage.
- Ask follow-up questions with the recent conversation included as context.
- Copy responses, export a conversation as Markdown, and give helpful/not-helpful
  feedback.
- Switch between light and dark themes. Keyboard shortcuts: `Ctrl+K` to find
  conversations, `Ctrl+N` to start a new one, and `Enter` to send.

## Run on Windows

Open PowerShell in the repository folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
code .env
```

In `.env`, add a Groq API key:

```dotenv
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Create a key at [Groq Console](https://console.groq.com/keys). Keep `.env`
private; it is ignored by Git. Groq receives the question and retrieved text
passages to generate each answer. The source PDFs and Chroma database stay on
your computer.

Start the website:

```powershell
.\.venv\Scripts\python.exe -m full_pipeline.app
```

Open **http://127.0.0.1:5000** in your browser. The first question loads the
local embedding model if it is not already cached. Use the message box to ask a
question; Smash AI will search the existing Chroma index and answer in the chat.
Conversation history is saved only in this browser. Use the delete control in
the sidebar to remove individual chats.

## Add or re-index PDFs

Put additional PDFs in `data/user_docs/`. The folder is local and its PDFs are
ignored by Git. Rebuild the index from the included PDFs and your added PDFs:

```powershell
.\.venv\Scripts\python.exe -m full_pipeline.pipeline
```

This uses the existing document loader, 500-token chunker with 100-token
overlap, sentence-transformer embedder, and Chroma store. Chunks are also saved
to `data/processed/chunks.jsonl`.

Run the focused tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Project layout

```text
full_pipeline/app.py           Local Flask server and chat API
templates/index.html            Smash AI chat page
static/styles.css               Website styling
static/app.js                   Chat interactions
full_pipeline/pipeline.py      Retrieval, answer generation, PDF indexing
vector_db/chroma/               Existing Chroma vector-store implementation
data/processed/                 Extracted documents and chunks
data/vector_db/chroma/          Persistent local vector database
data/user_docs/                 Your local PDFs (Git-ignored)
```

## GitHub

The changes can be committed as one focused interface update:

```powershell
git add full_pipeline/app.py prompting/prompt_builder.py requirements.txt README.md
git add templates static tests
git add -u -- advanced/advanced_rag.py data/download_samples.py evaluation/evaluator.py retrieval/retriever.py requirements-minimal.txt config.yaml
git commit -m "Replace Streamlit with Smash AI Flask chat"
git push -u origin HEAD
```

Repository: https://github.com/Sreenayak/RAG_Own_Decision_Making_System
