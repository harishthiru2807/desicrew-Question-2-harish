"""Streamlit entry point for the Document-Aware AI Support Assistant."""

from __future__ import annotations

import streamlit as st

from config import AppConfig, get_config
from conversation_memory import ConversationMemory
from demo import DEMO_EXCHANGES, run_demonstration, sample_document_texts
from embedding_service import SentenceTransformerEmbeddingService
from runtime import AssistantRuntime

st.set_page_config(
    page_title="Document-Aware AI Support Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px; }
    .app-title { font-size: 1.7rem; font-weight: 700; color: #1F4E79; margin-bottom: 0.2rem; }
    .app-subtitle { color: #3D4C5F; font-size: 0.98rem; margin-bottom: 0.8rem; }
    .status-chip {
        display: inline-block; background: #E8EEF5; color: #1F4E79;
        border: 1px solid #C5D2E3; border-radius: 999px; padding: 0.25rem 0.75rem;
        font-size: 0.85rem; margin-bottom: 0.8rem;
    }
    .empty-kb {
        background: #F3F6FA; border: 1px dashed #B7C6D9; border-radius: 12px;
        padding: 1.2rem; color: #3D4C5F;
    }
    [data-testid="stChatMessage"] { border-radius: 12px; }
    .privacy-note { font-size: 0.82rem; color: #5B6B7C; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _init_state() -> None:
    if "config" not in st.session_state:
        st.session_state.config = get_config()
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "runtime" not in st.session_state:
        st.session_state.runtime = None
        st.session_state.embed_error = None
    if "confirm_clear_kb" not in st.session_state:
        st.session_state.confirm_clear_kb = False
    if "demo_run" not in st.session_state:
        st.session_state.demo_run = None
    if "replace_uploads" not in st.session_state:
        st.session_state.replace_uploads = False
    if "welcome_shown" not in st.session_state:
        st.session_state.welcome_shown = True


@st.cache_resource(show_spinner=False)
def _cached_sentence_embeddings(model_name: str) -> SentenceTransformerEmbeddingService:
    # The transformer weights download on the first embed() call, which needs internet access.
    return SentenceTransformerEmbeddingService(model_name)


def get_runtime() -> AssistantRuntime | None:
    if st.session_state.runtime is not None:
        return st.session_state.runtime
    cfg: AppConfig = st.session_state.config
    try:
        embeddings = _cached_sentence_embeddings(cfg.embedding_model)
        st.session_state.runtime = AssistantRuntime(embeddings=embeddings, config=cfg)
        st.session_state.embed_error = None
        return st.session_state.runtime
    except Exception:
        st.session_state.embed_error = (
            "The embedding model could not be loaded. The first run needs internet access "
            f"to download '{cfg.embedding_model}'. Check the network connection and try again."
        )
        return None


def _sync_runtime_config(runtime: AssistantRuntime) -> None:
    cfg: AppConfig = st.session_state.config
    cfg.validate_chunking()
    runtime.config = cfg
    runtime.retriever.min_score = cfg.min_relevance_score
    runtime.llm_client.config = cfg
    runtime.topic_manager.llm_client = runtime.llm_client


def render_citations(answer) -> None:
    if not answer.citations:
        return
    st.markdown("**Citations**")
    for citation in answer.citations:
        with st.expander(citation.format_label(), expanded=False):
            st.write(f"**Filename:** {citation.filename}")
            st.write(f"**Page:** {citation.page_number if citation.page_number is not None else 'Not available'}")
            st.write(f"**Section:** {citation.section_heading or 'Not available'}")
            st.write(f"**Chunk:** {citation.chunk_id}")
            if citation.score is not None:
                st.caption(
                    f"Similarity rank score: {citation.score:.3f} (not a probability of factual correctness)."
                )
            st.markdown("**Exact supporting passage**")
            st.write(citation.passage)


def render_header(runtime: AssistantRuntime | None) -> None:
    st.markdown('<div class="app-title">Document-Aware AI Support Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Upload policies and guides, then ask questions. '
        "Answers are grounded in retrieved passages with inspectable citations.</div>",
        unsafe_allow_html=True,
    )
    status = runtime.knowledge_status() if runtime else "Embedding model is not ready."
    st.markdown(f'<div class="status-chip">{status}</div>', unsafe_allow_html=True)
    cfg: AppConfig = st.session_state.config
    if not cfg.llm_configured:
        st.info(
            "No Gemini API key is configured. You can paste your Gemini API key in the sidebar settings, "
            "or set `GEMINI_API_KEY` in `.env` or Streamlit secrets. "
            "Upload, indexing, and retrieval remain fully operational in evidence-only mode until a key is set."
        )


def render_sidebar(runtime: AssistantRuntime | None) -> None:
    cfg: AppConfig = st.session_state.config
    st.sidebar.header("Knowledge base")
    uploads = st.sidebar.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Files stay in this session. Relevant passages are sent to the Gemini API only when generating answers.",
    )
    st.session_state.replace_uploads = st.sidebar.checkbox(
        "Replace files with the same name",
        value=st.session_state.replace_uploads,
    )
    process = st.sidebar.button("Process documents", type="primary", use_container_width=True, disabled=runtime is None)

    if process and runtime and uploads:
        _sync_runtime_config(runtime)
        for uploaded in uploads:
            data = uploaded.getvalue()
            with st.sidebar.status(f"Processing {uploaded.name}", expanded=False) as status:
                result = runtime.process_upload(data, uploaded.name, replace=st.session_state.replace_uploads)
                status.update(label=f"{uploaded.name}: {result.status}", state="complete" if result.status == "indexed" else "error")
            if result.status == "indexed":
                st.sidebar.success(f"{result.filename}: {result.message}")
            elif result.status == "duplicate":
                st.sidebar.warning(result.message)
            elif result.status == "needs_ocr":
                st.sidebar.error(result.message)
            else:
                st.sidebar.error(result.message)
        st.rerun()

    st.sidebar.subheader("Uploaded documents")
    if runtime is None or not runtime.store.documents:
        st.sidebar.caption("No documents indexed yet.")
    else:
        for doc in list(runtime.store.documents.values()):
            cols = st.sidebar.columns([4, 1])
            cols[0].markdown(f"**{doc.filename}**  \n{doc.status} · {doc.chunk_count} chunks")
            if cols[1].button("✕", key=f"rm_{doc.document_id}", help=f"Remove {doc.filename}"):
                runtime.remove_document(doc.document_id)
                st.rerun()
            with st.sidebar.expander(f"Preview {doc.filename}", expanded=False):
                st.write(doc.preview_text or "No preview available.")

    if st.sidebar.button("Rebuild index", use_container_width=True, disabled=runtime is None):
        if runtime:
            runtime.rebuild_index()
            st.sidebar.success("Index rebuilt from the current documents.")

    if st.sidebar.button("Clear knowledge base", use_container_width=True, disabled=runtime is None):
        st.session_state.confirm_clear_kb = True
    if st.session_state.confirm_clear_kb:
        st.sidebar.warning("This removes all indexed documents from this session.")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Confirm clear", type="primary"):
            if runtime:
                runtime.clear_knowledge_base()
            st.session_state.confirm_clear_kb = False
            st.rerun()
        if c2.button("Cancel"):
            st.session_state.confirm_clear_kb = False
            st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Conversation")
    if st.sidebar.button("New chat", use_container_width=True):
        st.session_state.memory = ConversationMemory()
        st.rerun()
    if st.sidebar.button("Clear conversation", use_container_width=True):
        st.session_state.memory.clear_conversation()
        st.rerun()
    st.sidebar.caption("New chat and clear conversation keep the knowledge base. Sessions do not share chat history.")

    st.sidebar.divider()
    st.sidebar.header("Gemini & RAG Settings")

    gemini_key_input = st.sidebar.text_input(
        "Gemini API Key",
        value=cfg.gemini_api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Google Gemini API key from Google AI Studio. Set in sidebar or in .env / secrets as GEMINI_API_KEY.",
    )
    if gemini_key_input != cfg.gemini_api_key:
        cfg.gemini_api_key = gemini_key_input
        if runtime:
            _sync_runtime_config(runtime)
        st.rerun()

    models_options = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.5-flash", "Custom"]
    cur_idx = models_options.index(cfg.gemini_model) if cfg.gemini_model in models_options else len(models_options) - 1
    chosen_model = st.sidebar.selectbox("Gemini Model", models_options, index=cur_idx)
    if chosen_model == "Custom":
        custom_name = st.sidebar.text_input("Custom model name", value=cfg.gemini_model)
        if custom_name != cfg.gemini_model:
            cfg.gemini_model = custom_name
            if runtime:
                _sync_runtime_config(runtime)
    else:
        if chosen_model != cfg.gemini_model:
            cfg.gemini_model = chosen_model
            if runtime:
                _sync_runtime_config(runtime)

    if cfg.llm_configured:
        st.sidebar.success(f"Gemini API key active ({cfg.gemini_model})")
    else:
        st.sidebar.warning("Gemini key missing — evidence-only answers.")

    cfg.chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 2000, cfg.chunk_size, 50)
    cfg.chunk_overlap = st.sidebar.slider("Chunk overlap (characters)", 0, 400, cfg.chunk_overlap, 10)
    cfg.top_k = st.sidebar.slider("Top-k retrieval", 1, 12, cfg.top_k, 1)
    cfg.min_relevance_score = st.sidebar.slider("Minimum similarity filter", 0.0, 0.8, float(cfg.min_relevance_score), 0.01)
    cfg.embedding_model = st.sidebar.text_input("Embedding model", value=cfg.embedding_model, disabled=True)
    st.sidebar.caption("Embeddings run locally with Sentence Transformers. The first download needs internet access.")

    st.sidebar.divider()
    st.sidebar.header("Demonstration Mode")
    st.sidebar.caption("Runs ten live exchanges against the sample policy and product guide.")
    if st.sidebar.button("Run 10-exchange demonstration", use_container_width=True, disabled=runtime is None):
        if runtime is not None:
            with st.spinner("Running demonstration through the live pipeline…"):
                st.session_state.demo_run = run_demonstration(runtime=runtime, memory=ConversationMemory())
            st.rerun()


def render_demo_panel() -> None:
    demo = st.session_state.demo_run
    if demo is None:
        return
    st.subheader("Demonstration Mode")
    st.write(demo.status_message)
    st.write(f"**Status:** {'Executed through the live pipeline' if demo.executed else 'Not executed'}")
    st.write(f"**Generated answers:** {'Yes' if demo.generated_answers else 'No — LLM not used'}")
    st.write(f"**Exchanges:** {demo.completed_count} / {demo.total}")
    st.write("**Sample source documents:** " + ", ".join(demo.indexed_files))
    with st.expander("Read sample source documents", expanded=False):
        for name, text in sample_document_texts().items():
            st.markdown(f"**{name}**")
            st.text(text)
    for item in demo.exchanges:
        st.markdown(f"#### Exchange {item.spec.number} / {demo.total}: {item.spec.label}")
        with st.chat_message("user"):
            st.write(item.question)
        with st.chat_message("assistant"):
            if item.answer:
                st.write(item.answer.text)
                render_citations(item.answer)
                if item.notes:
                    st.warning(item.notes)
            else:
                st.error("This exchange did not run.")
    st.divider()


def render_chat(runtime: AssistantRuntime | None) -> None:
    memory: ConversationMemory = st.session_state.memory
    if runtime is None:
        st.error(st.session_state.embed_error or "The assistant runtime is unavailable.")
        return

    if not memory.messages:
        st.markdown(
            '<div class="empty-kb"><strong>Welcome.</strong> Upload documents in the sidebar, '
            "process them, then ask questions here. Follow-ups, topic changes, and recap requests "
            "are supported. Factual claims stay tied to retrieved passages.</div>",
            unsafe_allow_html=True,
        )
        if runtime.store.is_empty:
            st.caption("The knowledge base is empty. Retrieval cannot run until at least one document is indexed.")

    for message in memory.messages:
        with st.chat_message(message.role):
            st.write(message.content)
            if message.role == "assistant" and message.citations:
                dummy = type("Answer", (), {"citations": message.citations})()
                render_citations(dummy)
            if message.follow_ups:
                st.caption("Suggested follow-ups: " + " · ".join(message.follow_ups))

    prompt = st.chat_input("Ask a question about your uploaded documents")
    if prompt:
        if runtime.store.is_empty:
            st.warning("Index at least one document before asking questions.")
            return
        _sync_runtime_config(runtime)
        with st.status("Retrieving evidence and preparing an answer…", expanded=False) as status:
            runtime.ask(prompt, memory)
            status.update(label="Answer ready", state="complete")
        st.rerun()

    st.markdown(
        '<p class="privacy-note">Documents and indexes stay in this browser session by default. '
        "When generated answers are enabled, relevant retrieved passages are sent to the configured LLM provider. "
        "API keys are never shown in the interface. Similarity scores are not calibrated probabilities.</p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    _init_state()
    # Streamlit Cloud secrets
    try:
        for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY"):
            if key in st.secrets and st.secrets[key]:
                st.session_state.config.llm_api_key = str(st.secrets[key]).strip()
                break
        for key in ("GEMINI_MODEL", "LLM_MODEL"):
            if key in st.secrets and st.secrets[key]:
                st.session_state.config.llm_model = str(st.secrets[key]).strip()
                break
        if "LLM_BASE_URL" in st.secrets and st.secrets["LLM_BASE_URL"]:
            st.session_state.config.llm_base_url = str(st.secrets["LLM_BASE_URL"]).strip()
    except Exception:
        pass

    runtime = get_runtime()
    render_header(runtime)
    render_sidebar(runtime)
    render_demo_panel()
    render_chat(runtime)


if __name__ == "__main__":
    main()
