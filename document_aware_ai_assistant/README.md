# Document-Aware AI Support Assistant

Standalone Streamlit application for **DesiCrew Solutions Data Science Internship Evaluation**. Users upload PDF, DOCX, and TXT files, ask questions, and receive evidence-based answers with citations that can be inspected. Conversational context is kept inside the current Streamlit session only.

## Architecture

```
Upload → extract (PyMuPDF / python-docx / TXT) → chunk → embed (Sentence Transformers)
      → FAISS index → retrieve + filter → optional LLM grounded generation → validated citations
```

| Module | Role |
| --- | --- |
| `app.py` | Streamlit UI, session isolation, demonstration panel |
| `config.py` | Environment and runtime settings |
| `document_loader.py` | File validation and text extraction |
| `text_chunker.py` | Overlapping chunks with source metadata |
| `embedding_service.py` | Local embeddings; deterministic double for tests |
| `vector_store.py` | FAISS index plus document lifecycle |
| `retriever.py` | Top-k search and relevance filtering |
| `answer_generator.py` | Grounded answers or evidence-only mode |
| `citations.py` | Citation formatting and validation |
| `conversation_memory.py` | History, recap, entities, topics |
| `topic_manager.py` | Follow-ups, topic switch, fallback if no LLM |
| `runtime.py` | Orchestration used by the app, demo, and tests |
| `demo.py` | Ten live exchanges against sample documents |

Similarity scores rank passages. They are **not** calibrated probabilities of factual correctness.

## Requirements

- Python 3.11 (or another supported 3.10+ version)
- Windows PowerShell for the commands below
- Internet access on the **first** embedding-model download (`sentence-transformers/all-MiniLM-L6-v2`)
- Google Gemini API key (from [Google AI Studio](https://aistudio.google.com/))

## Installation (Windows PowerShell)

From this directory (`document_aware_ai_assistant`):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally with your Google Gemini credentials. Do not commit it.

```
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Supported Gemini models include `gemini-1.5-flash` (recommended default), `gemini-2.0-flash`, and `gemini-1.5-pro`. You can also enter or switch your Gemini API key directly in the Streamlit sidebar settings at runtime.

## Run the application

```powershell
streamlit run app.py
```

The UI uses a clean light theme. Upload one or more PDF, DOCX, or TXT files, click **Process documents**, then ask questions in the main chat.

## Run tests

Tests do **not** need an external API key or network access. Embeddings in unit tests are deterministic so transformer weights are not downloaded, and Gemini API calls are thoroughly unit-tested via mocks.

```powershell
python -m pytest -q
```

## Demonstration Mode

The sidebar action **Run 10-exchange demonstration** indexes:

- `demo_documents/employee_onboarding_leave_policy.txt`
- `demo_documents/product_support_troubleshooting_guide.txt`

It then executes ten consecutive turns through the **same** extraction, chunking, indexing, retrieval, and answer pipeline as the main chat. It is not a hardcoded transcript.

If `GEMINI_API_KEY` is not provided, the demonstration still retrieves real passages and citations in evidence-only mode. Once `GEMINI_API_KEY` is provided, grounded answers are synthesized by Gemini.

The ten exchanges cover: a direct question, a follow-up, a clarification, a topic switch, a return to the earlier topic, an uncovered question, a previously answered question, a recap, a cross-document comparison, and a final context-retention question.

## Behaviour notes

- Empty files and PDFs with no extractable text are **not** indexed.
- Image-only PDFs show that they may need OCR. **OCR is not performed.**
- Duplicate filenames are rejected unless **Replace files with the same name** is enabled.
- Removing or replacing a document rebuilds the index without its old chunks.
- **Clear conversation** and **New chat** keep the knowledge base.
- Sessions do not share chat history (`st.session_state`).
- Citations are built from retrieved chunk metadata, then validated. The LLM is not trusted to invent sources.
- When evidence is insufficient, the assistant replies:

  `I could not find sufficient information to answer this question in your uploaded knowledge base.`

## Security and privacy

- Uploads and indexes stay in the current session memory by default.
- Relevant retrieved passages are sent to Google Gemini only when generated answers are requested.
- API keys are read from `.env`, Streamlit secrets, or the password input in the sidebar, and are never rendered in plain text in the UI.
- File size is limited (10 MB per file). Unsupported types are rejected.
- Application logs do not print API keys or full document dumps.

## Streamlit Community Cloud deployment

1. Push this folder (or the repository that contains it) to GitHub without `.env`.
2. Create an app with entry point `document_aware_ai_assistant/app.py` (or `app.py` if this folder is the repo root).
3. In **Settings → Secrets**, add:

```toml
GEMINI_API_KEY = "AIzaSy..."
GEMINI_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

4. Confirm `requirements.txt` is present. `faiss-cpu`, `pymupdf`, and `sentence-transformers` install from wheels on Cloud in typical cases. The first run downloads the embedding model; allow several minutes.
5. Do not upload `.streamlit/secrets.toml` to git.

## Limitations

- Scanned PDFs need an external OCR step before upload.
- Heading detection is heuristic.
- Deterministic test embeddings are not production-quality semantic vectors.
- Without an LLM key, the app retrieves and displays evidence but does not synthesize a narrative answer.
- Very large corpora may need a persistence layer beyond session memory.

## Project layout

See the module table above plus `tests/`, `demo_documents/`, `.env.example`, and `.gitignore`.
