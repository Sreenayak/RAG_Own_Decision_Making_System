# Grounded - RAG on your own documents

Ask questions about your PDFs and get answers grounded in retrieved passages,
with citations back to the source file and page. The interface also has an
optional side-by-side comparison against the same LLM without document retrieval.

## What is included

- Ten included PDFs with about 220 pages of extracted text.
- Sentence-transformer embeddings stored in the persistent local ChromaDB index.
- 307 existing passages, chunked at 500 tokens with 100-token overlap.
- A Streamlit chat interface with source passages, similarity scores, PDF uploads,
  and a one-click local re-index action.
- Groq chat completions for answer generation. Only the question and retrieved
  passages are sent to Groq; PDFs and the Chroma database remain local.

## Run on Windows

Use Python 3.10 or newer. From the repository directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Create a Groq API key at [Groq Console](https://console.groq.com/keys), then edit
`.env` and replace the placeholder:

```dotenv
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Keep `.env` private; it is ignored by Git. Alternatively, enter the key in the
app's password field. The model name can be changed to any currently available
Groq chat model. See the [Groq model list](https://console.groq.com/docs/models)
for current availability and pricing.

Start the app:

```powershell
python -m streamlit run full_pipeline/app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). The first query
loads the local embedding model; first-time model downloads need an internet
connection. Ask a sample question, inspect the cited source passages, or use the
sidebar to upload your own PDFs. Uploaded PDFs are saved under `data/user_docs/`
and are ignored by Git. Click **Save & rebuild local index** to include them.
For local use only, do not upload documents you are not allowed to share: the
retrieved excerpts are sent to Groq when you ask a question.

To re-index the PDFs already in `data/raw/` and `data/user_docs/` from a terminal:

```powershell
python -m full_pipeline.pipeline
```

The resulting chunks and vectors are persisted in `data/processed/` and
`data/vector_db/chroma/`.

Run the focused tests with:

```powershell
python -m unittest discover -s tests -v
```

## Retrieval and grounding example

The indexed RAG PDF contains this passage on page 3:

> Pre-training the retriever using the Inverse Cloze Task (ICT), a technique
> that helps the model learn retrieval patterns by predicting masked text
> within documents.

Try this question in the app:

> What technique predicts masked text within documents to pre-train a retriever?

The RAG answer should identify the **Inverse Cloze Task (ICT)** and cite
`retrieval_augmented_generation.pdf`, page 3. The dense retriever was run against
this question during development and ranked that PDF page first (cosine
similarity 0.526).

Turn on **Compare with plain LLM** in the sidebar to make an additional call
without passing any document excerpts. To demonstrate the difference with your
own material, upload a private or organization-specific PDF and ask about a detail
that is not publicly known. The plain call has no access to that file; the RAG
answer can retrieve and cite it. Model responses may vary, so use the source
panel to verify the answer rather than treating either response as ground truth.

## Project layout

```text
full_pipeline/app.py           Streamlit chat UI
full_pipeline/pipeline.py      Retrieval, answer generation, PDF indexing
llm_integration/llm_client.py  Groq API client
prompting/prompt_builder.py   Citation-aware grounding prompt
data/raw/                      Included PDF documents
data/user_docs/                Local user PDFs (Git-ignored)
data/vector_db/chroma/         Persistent local vector database
```

## Git commits

The implementation is grouped into three main changes. From PowerShell, stage
and commit each group separately:

```powershell
# 1. Grounded answer pipeline and Groq integration
git rm --cached --ignore-unmatch .env
git add full_pipeline/pipeline.py llm_integration/llm_client.py prompting/prompt_builder.py .env.example .gitignore
git commit -m "Add grounded RAG answer pipeline"

# 2. Chat interface and local PDF upload/indexing
git add full_pipeline/app.py data/user_docs/.gitkeep
git commit -m "Add Streamlit document chat interface"

# 3. Setup instructions, dependencies, and focused tests
git add README.md requirements.txt tests
git commit -m "Document and test the RAG workflow"
```

Then publish the commits to your GitHub repository:

```powershell
git push -u origin HEAD
```

Do not commit `.env` or private PDFs. This checkout previously tracked an empty
`.env`; the implementation removes it from Git's index while preserving the
local file. The repository already has
a GitHub remote; check that the repository visibility is public in GitHub
settings if you need to submit a public URL.