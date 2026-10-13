# DesiCrew Solutions - Document-Aware AI Support Assistant

A standalone, production-ready Streamlit application built for the **DesiCrew Solutions Data Science Internship Evaluation**.

The application allows users to upload documents (PDF, DOCX, TXT), ask questions about their contents, receive evidence-based answers with inspectable citations, and maintain multi-turn conversational context.

## Project Structure

All source code, tests, demonstration documents, and application configurations are located inside the [`document_aware_ai_assistant/`](./document_aware_ai_assistant) directory:

- **[`document_aware_ai_assistant/app.py`](./document_aware_ai_assistant/app.py)**: Streamlit UI entry point.
- **[`document_aware_ai_assistant/config.py`](./document_aware_ai_assistant/config.py)**: Configuration and environment settings (Google Gemini API, chunk sizes, thresholds).
- **[`document_aware_ai_assistant/document_loader.py`](./document_aware_ai_assistant/document_loader.py)**: Text extraction for PDF, DOCX, and TXT with heading detection.
- **[`document_aware_ai_assistant/text_chunker.py`](./document_aware_ai_assistant/text_chunker.py)**: Overlapping text chunker preserving document and page metadata.
- **[`document_aware_ai_assistant/embedding_service.py`](./document_aware_ai_assistant/embedding_service.py)**: Sentence Transformers embedding service.
- **[`document_aware_ai_assistant/vector_store.py`](./document_aware_ai_assistant/vector_store.py)**: FAISS-CPU vector indexing and document lifecycle manager.
- **[`document_aware_ai_assistant/retriever.py`](./document_aware_ai_assistant/retriever.py)**: Semantic retrieval with relevance threshold filtering.
- **[`document_aware_ai_assistant/answer_generator.py`](./document_aware_ai_assistant/answer_generator.py)**: Grounded RAG answer generator with Google Gemini and evidence-only fallback.
- **[`document_aware_ai_assistant/citations.py`](./document_aware_ai_assistant/citations.py)**: Citation formatting and validation against indexed chunks.
- **[`document_aware_ai_assistant/conversation_memory.py`](./document_aware_ai_assistant/conversation_memory.py)**: Multi-turn session memory and topic tracking.
- **[`document_aware_ai_assistant/topic_manager.py`](./document_aware_ai_assistant/topic_manager.py)**: Follow-ups, topic-switching, and clarification handler.
- **[`document_aware_ai_assistant/demo.py`](./document_aware_ai_assistant/demo.py)**: 10-exchange automated demonstration pipeline.
- **[`document_aware_ai_assistant/tests/`](./document_aware_ai_assistant/tests)**: 46 automated unit and integration tests.

## Quick Start (Windows PowerShell)

```powershell
cd document_aware_ai_assistant
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

For full documentation and Streamlit Cloud deployment instructions, see [`document_aware_ai_assistant/README.md`](./document_aware_ai_assistant/README.md).
