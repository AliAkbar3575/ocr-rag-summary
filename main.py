import streamlit as st
import tempfile
import os
from pathlib import Path
from utils import save_as_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="DocOCR Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1.5rem 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 24px; font-weight: 500; }
    section[data-testid="stSidebar"] { min-width: 320px; }
    .block-container { max-width: 1100px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📄 Handwritten Document OCR Intelligence</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #6c757d; margin-bottom: 2rem;'>"
    "Upload a scanned handwritten document (image or PDF) — extract text, generate summaries, or ask questions with RAG.</p>",
    unsafe_allow_html=True,
)

# ---- API key: Streamlit Cloud secrets > .env file ----
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = None
if not GROQ_API_KEY:
    from dotenv import load_dotenv, find_dotenv
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
else:
    st.warning(
        "🔑 **GROQ_API_KEY** not set.\n\n"
        "**Local:** Create a `.env` file with:\n"
        "```\nGROQ_API_KEY=your_key_here\n```\n\n"
        "**Streamlit Cloud:** Add `GROQ_API_KEY` in Settings → Secrets.",
        icon="⚠️",
    )

# ---- ensure output directory exists ----
os.makedirs("./output", exist_ok=True)

# ---- available models ----
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

# ---- session state ----
DEFAULT_STATE = {
    "ocr_text": None,
    "ocr_text_list": None,
    "processed": False,
    "rag_answer": None,
    "rag_sources": None,
    # Summary HITL state
    "summary_phase": "idle",       # idle | reviewing | approved | error
    "summary_gen": None,           # SummaryGenerator instance
    "summary_draft": None,         # current summary being reviewed
    "summary_final": None,         # final approved summary
}
for k, v in DEFAULT_STATE.items():
    st.session_state.setdefault(k, v)


@st.cache_resource(show_spinner="Loading embedding model (once)...")
def _load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ======================== SIDEBAR ========================
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a scanned document",
        type=["jpg", "jpeg", "png", "pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        ext = Path(uploaded_file.name).suffix.lower()
        size_kb = uploaded_file.size / 1024
        st.caption(f"📎 `{uploaded_file.name}`  ·  {size_kb:.1f} KB")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
        tmp.close()

        st.divider()
        st.selectbox(
            "LLM Model",
            options=AVAILABLE_MODELS,
            index=0,
            key="selected_model",
            help="Model used for summary generation and RAG querying.",
        )

        if st.button("🔍 Process Document", type="primary", use_container_width=True):
            for k in DEFAULT_STATE:
                st.session_state[k] = DEFAULT_STATE[k]

            from PIL import Image
            from utils import pdf_to_images, image_to_base64, ocr_page

            bar = st.progress(0, text="Initialising…")
            try:
                if ext in (".jpg", ".jpeg", ".png"):
                    bar.progress(30, text="Running OCR on image…")
                    pil_img = Image.open(tmp_path)
                    b64 = image_to_base64(pil_img)
                    ocr_result = ocr_page(b64)
                    text_list = [ocr_result]
                    text = ocr_result
                    bar.progress(95, text="Saving…")
                else:
                    bar.progress(5, text="Converting PDF to images…")
                    images = pdf_to_images(tmp_path)
                    text_list = []
                    total = len(images)
                    for i, img in enumerate(images, 1):
                        pct = int(5 + 75 * i / total)
                        bar.progress(pct, text=f"OCR on page {i}/{total}…")
                        b64 = image_to_base64(img)
                        t = ocr_page(b64)
                        text_list.append(f"\n--- Page {i} ---\n{t}")
                    text = "\n".join(text_list)
                    bar.progress(95, text="Saving…")

                save_as_text(text, "./output/extracted_text.txt")
                st.session_state.ocr_text = text
                st.session_state.ocr_text_list = text_list
                st.session_state.processed = True
                bar.progress(100, text="Done!")
            except Exception as exc:
                bar.empty()
                st.error(f"❌ Processing failed: {exc}")
            finally:
                os.unlink(tmp_path)

        if st.session_state.processed:
            st.divider()
            st.caption(f"**Extracted:** {len(st.session_state.ocr_text):,} chars")
            if st.button("🔄 Reset", use_container_width=True):
                for k in DEFAULT_STATE:
                    st.session_state[k] = DEFAULT_STATE[k]
                st.rerun()
    else:
        st.info("Upload a JPG, PNG, or PDF file")
        st.page_link("https://groq.com", label="Get a Groq API key →", icon="🔑")

# ======================== TABS ========================
tab_summary, tab_rag = st.tabs(["📝 Summarisation", "🔍 RAG Query"])

# ######################## SUMMARY TAB ########################
with tab_summary:
    if not st.session_state.processed:
        st.info("👆 Upload and process a document using the sidebar first.")
    else:
        with st.expander("📄 Extracted Text", expanded=False):
            st.text_area("raw_ocr_summary", st.session_state.ocr_text, height=280, label_visibility="collapsed")

        st.markdown("### ✨ Generate Summary")
        model_display = st.session_state.get("selected_model", "llama-3.3-70b-versatile")
        st.caption(f"Model: **{model_display}** · Human-in-the-loop LangGraph workflow.")

        phase = st.session_state.summary_phase

        # ------------ IDLE ------------
        if phase == "idle":
            if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
                from summary_gen.summary2 import SummaryGenerator
                from langgraph.types import Command

                sg = SummaryGenerator(
                    st.session_state.ocr_text,
                    thread_id=st.session_state.get("session_id", "summary-thread"),
                    model_name=st.session_state.selected_model,
                )
                try:
                    result = sg.engine.app.invoke(
                        {"input_text": st.session_state.ocr_text},
                        config=sg.config,
                    )
                except Exception as exc:
                    st.error(f"❌ Summary generation failed: {exc}")
                    st.session_state.summary_phase = "error"
                    st.rerun()

                if "__interrupt__" in result:
                    st.session_state.summary_gen = sg
                    st.session_state.summary_draft = (
                        result["__interrupt__"][0].value.get("case_fact_summary", "")
                    )
                    st.session_state.summary_phase = "reviewing"
                else:
                    st.session_state.summary_final = result.get("case_fact_summary", "")
                    st.session_state.summary_phase = "approved"
                    save_as_text(st.session_state.summary_final, "./output/summary_output.txt")
                st.rerun()

        # ------------ REVIEWING (HITL loop) ------------
        if phase == "reviewing":
            sg = st.session_state.summary_gen
            if sg is None:
                st.error("Session state error — please generate again.")
                st.session_state.summary_phase = "idle"
                st.rerun()

            st.info("📝 **Review the draft below** — approve it or give feedback to revise.")
            st.markdown("**Draft Summary:**")
            st.markdown(
                f'<div style="background: #1e1e2e; color: #e0e0e0; padding: 1rem 1.5rem; border-radius: 10px; '
                f'border-left: 4px solid #f0c040; line-height: 1.6; margin-bottom: 1.5rem;">'
                f'{st.session_state.summary_draft}</div>',
                unsafe_allow_html=True,
            )

            from langgraph.types import Command

            col_app, col_fb = st.columns([1, 3])
            with col_app:
                if st.button("✅ Approve", type="primary", use_container_width=True):
                    try:
                        result = sg.engine.app.invoke(
                            Command(resume="yes"),
                            config=sg.config,
                        )
                    except Exception as exc:
                        st.error(f"❌ Approval failed: {exc}")
                        st.session_state.summary_phase = "idle"
                        st.rerun()

                    if "__interrupt__" in result:
                        st.session_state.summary_draft = (
                            result["__interrupt__"][0].value.get("case_fact_summary", "")
                        )
                    else:
                        st.session_state.summary_final = result.get("case_fact_summary", "")
                        st.session_state.summary_phase = "approved"
                        st.session_state.summary_gen = None
                        save_as_text(st.session_state.summary_final, "./output/summary_output.txt")
                    st.rerun()

            with col_fb:
                feedback = st.text_area(
                    "Feedback for revision",
                    placeholder="e.g. Make it shorter, change the date, add more detail…",
                    label_visibility="collapsed",
                    key="feedback_input",
                )

            if st.button("🔄 Revise with Feedback", use_container_width=True):
                if not feedback.strip():
                    st.warning("Enter feedback or use Approve instead.")
                else:
                    try:
                        result = sg.engine.app.invoke(
                            Command(resume=feedback.strip()),
                            config=sg.config,
                        )
                    except Exception as exc:
                        st.error(f"❌ Revision failed: {exc}")
                        st.session_state.summary_phase = "idle"
                        st.rerun()

                    if "__interrupt__" in result:
                        st.session_state.summary_draft = (
                            result["__interrupt__"][0].value.get("case_fact_summary", "")
                        )
                    else:
                        st.session_state.summary_final = result.get("case_fact_summary", "")
                        st.session_state.summary_phase = "approved"
                        st.session_state.summary_gen = None
                        save_as_text(st.session_state.summary_final, "./output/summary_output.txt")
                    st.rerun()

            if st.button("🔄 Start Over (discard draft)", use_container_width=True):
                st.session_state.summary_phase = "idle"
                st.session_state.summary_draft = None
                st.session_state.summary_gen = None
                st.rerun()

        # ------------ APPROVED ------------
        if phase == "approved" and st.session_state.summary_final:
            st.success("✅ Summary approved!")
            st.markdown("### 📋 Final Summary")
            st.markdown(
                f'<div style="background: #1e1e2e; color: #e0e0e0; padding: 1rem 1.5rem; border-radius: 10px; '
                f'border-left: 4px solid #22c55e; line-height: 1.6; margin-bottom: 1.5rem;">'
                f'{st.session_state.summary_final}</div>',
                unsafe_allow_html=True,
            )

            c_dl, c_new = st.columns([1, 1])
            with c_dl:
                st.download_button(
                    "📥 Download Summary",
                    data=st.session_state.summary_final,
                    file_name="summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with c_new:
                if st.button("🔄 New Summary", use_container_width=True):
                    st.session_state.summary_phase = "idle"
                    st.session_state.summary_draft = None
                    st.session_state.summary_final = None
                    st.session_state.summary_gen = None
                    st.rerun()

        # ------------ ERROR ------------
        if phase == "error":
            if st.button("🔄 Try Again", use_container_width=True):
                st.session_state.summary_phase = "idle"
                st.rerun()

# ######################## RAG TAB ########################
with tab_rag:
    if not st.session_state.processed:
        st.info("👆 Upload and process a document using the sidebar first.")
    else:
        with st.expander("📄 Extracted Text", expanded=False):
            st.text_area("raw_ocr_rag", st.session_state.ocr_text, height=280, label_visibility="collapsed")

        st.markdown("### 🔍 Ask a Question")
        st.caption(f"Model: **{st.session_state.get('selected_model', 'llama-3.3-70b-versatile')}**")

        query = st.text_input(
            "Question",
            placeholder="e.g., What are the main topics discussed?",
            label_visibility="collapsed",
        )

        ask_clicked = st.button("🔎 Ask", type="primary", use_container_width=True, disabled=not query)

        if query and ask_clicked:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_groq import ChatGroq
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document

            with st.spinner("⏳ Running RAG pipeline…"):
                try:
                    llm = ChatGroq(model=st.session_state.selected_model, temperature=0.7)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                    docs = []
                    for i, txt in enumerate(st.session_state.ocr_text_list, 1):
                        for chunk in splitter.split_text(txt):
                            docs.append(Document(page_content=chunk, metadata={"page": i}))

                    embeddings = _load_embeddings()
                    vs = FAISS.from_documents(docs, embeddings)
                    retriever = vs.as_retriever(search_kwargs={"k": 3})

                    retrieved = retriever.invoke(query)
                    context = "\n\n".join(d.page_content for d in retrieved)
                    prompt = f"""Answer the question using only the context below. Do not hallucinate.

Context:
{context}

Question:
{query}"""
                    response = llm.invoke(prompt)

                    st.session_state.rag_answer = response.content
                    st.session_state.rag_sources = [
                        {"page": d.metadata["page"], "snippet": d.page_content[:180]}
                        for d in retrieved
                    ]
                    save_as_text(
                        f"QUESTION: {query}\n\nANSWER: {response.content}\n\n"
                        + "\n".join(
                            f"Source (page {s['page']}): {s['snippet']}"
                            for s in st.session_state.rag_sources
                        ),
                        "./output/rag_output.txt",
                    )
                except Exception as exc:
                    st.error(f"❌ RAG query failed: {exc}")

        if st.session_state.rag_answer:
            st.markdown("---")
            st.markdown("### 💡 Answer")
            st.markdown(st.session_state.rag_answer)

            if st.session_state.rag_sources:
                with st.expander("📚 Source Chunks", expanded=False):
                    for i, src in enumerate(st.session_state.rag_sources, 1):
                        st.caption(f"**Source {i}** (Page {src['page']})")
                        st.code(src["snippet"], language="text")
