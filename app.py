from __future__ import annotations

import html
import pandas as pd
import streamlit as st

from auth_backend import init_auth_db, login_user, signup_user
from document_processing import parse_uploaded_file
from history_store import load_history, save_history
from offline_cache import get_offline_cache, is_offline_mode
from qa_engine import DocumentQA
from report_helpers import (
    build_report_pdf_bytes,
    build_report_text,
)
from ui_helpers import apply_theme, highlight_query_terms, keyword_snippets


st.set_page_config(page_title="AI Document Analysis System", page_icon="📄", layout="wide")
init_auth_db()


def init_state() -> None:
    defaults = {
        "history": load_history(),
        "doc_text": "",
        "doc_sections": [],
        "doc_name": "",
        "qa_engine": DocumentQA(),
        "last_result": None,
        "is_logged_in": False,
        "user_name": "",
        "active_nav": "Search",
        "search_history": [],
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 5,
        "show_retrieved_context": False,
        "show_llm_prompt": False,
        "show_performance_charts": False,
        "force_ocr_pdf": False,
        "offline_mode": is_offline_mode(),
        "cached_docs": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> tuple[bool, str]:
    with st.sidebar:
        st.markdown("## Workspace")
        dark_mode = st.toggle("Dark mode", value=True)
        
        # Offline mode indicator
        if st.session_state.offline_mode:
            st.info("🔌 **Offline Mode Active**")
        
        if not st.session_state.is_logged_in:
            auth_tab = st.radio("Account", ["Login", "Signup"], horizontal=True)
            if auth_tab == "Signup":
                name = st.text_input("Name", key="signup_name")
                email = st.text_input("Email", key="signup_email")
                password = st.text_input("Password", type="password", key="signup_password")
                if st.button("Create account"):
                    ok, msg = signup_user(name, email, password)
                    (st.success if ok else st.error)(msg)
            else:
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                if st.button("Login"):
                    ok, msg = login_user(email, password)
                    if ok:
                        st.session_state.is_logged_in = True
                        st.session_state.user_name = msg
                        st.success(f"Welcome, {msg}")
                        st.rerun()
                    else:
                        st.error(msg)
        else:
            st.success(f"Signed in as {st.session_state.user_name}")
            if st.button("Logout"):
                st.session_state.is_logged_in = False
                st.session_state.user_name = ""
                st.rerun()

        st.divider()
        nav_items = [
            "Search",
            "History",
            "Discover",
            "More Options",
            "Settings",
        ]
        for item in nav_items:
            if st.button(item, use_container_width=True, key=f"nav_{item}"):
                st.session_state.active_nav = item
        selected = st.session_state.active_nav

        st.markdown("### Upload history")
        if st.session_state.history:
            for item in reversed(st.session_state.history[-8:]):
                st.caption(f"- {item}")
        else:
            st.caption("No uploads yet.")
    return dark_mode, selected


def render_about_us() -> None:
    st.markdown("### About Us")
    st.markdown(
        """
        This project is an AI-powered Document Analysis System built with Python and Streamlit.
        It helps users upload `PDF`, `DOCX`, and `TXT` files, then ask natural language questions
        and get source-grounded answers quickly.

        **Core capabilities**
        - smart document parsing and text extraction
        - chunk-based retrieval with configurable settings
        - grounded Q&A with context and confidence score
        - summary generation and keyword search
        - multi-document comparison with export options
        - **offline document caching for working without internet**
        """
    )


def get_qa_config() -> dict:
    return {
        "chunk_size": int(st.session_state.chunk_size),
        "chunk_overlap": int(st.session_state.chunk_overlap),
        "top_k": int(st.session_state.top_k),
    }


def concise_text(text: str, max_chars: int = 220) -> str:
    clean = " ".join(text.split()).strip()
    if len(clean) <= max_chars:
        return clean
    short = clean[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{short}..."


def render_consistent_text(text: str) -> None:
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(f'<div class="answer-text">{safe}</div>', unsafe_allow_html=True)


def render_discover_details() -> None:
    st.markdown("## How It Works")
    cards = [
        ("📥", "Upload PDF", "Upload any PDF, DOCX, or TXT document such as notes, research papers, or reports."),
        ("🧩", "Chunking", "Text is split into overlapping chunks to preserve context across nearby sections."),
        ("🔢", "Embeddings", "Each chunk is vectorized with TF-IDF for fast semantic similarity search."),
        ("🗂️", "Index", "All chunk vectors are indexed in-memory for efficient retrieval during questions."),
        ("💬", "Ask a Question", "Your query is compared with chunk vectors to find the most relevant passages."),
        ("💡", "Answer Generation", "Top matching chunks are used to produce grounded answers with source labels."),
    ]
    for row_start in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, cards[row_start : row_start + 3]):
            with col:
                st.markdown(
                    f"""
                    <div class="card">
                        <h4>{icon} {title}</h4>
                        <p>{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_settings_panel() -> None:
    st.markdown("## Settings")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Retrieval Controls")
    left, right = st.columns(2)
    with left:
        st.session_state.chunk_size = st.slider(
            "Chunk Size (characters)", 200, 1500, st.session_state.chunk_size, 50
        )
        st.session_state.chunk_overlap = st.slider(
            "Chunk Overlap (characters)", 0, 500, st.session_state.chunk_overlap, 10
        )
    with right:
        st.session_state.top_k = st.slider("Top-K Retrieved Chunks", 1, 10, st.session_state.top_k, 1)
        st.caption("Higher Top-K increases context coverage but can slow responses.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Display Options")
    st.session_state.show_retrieved_context = st.toggle(
        "Show Retrieved Context", value=st.session_state.show_retrieved_context
    )
    st.session_state.show_llm_prompt = st.toggle("Show LLM Prompt", value=st.session_state.show_llm_prompt)
    st.session_state.show_performance_charts = st.toggle(
        "Show Performance Charts", value=st.session_state.show_performance_charts
    )
    st.session_state.force_ocr_pdf = st.toggle(
        "Force OCR for PDFs (recommended for scanned/broken text)",
        value=st.session_state.force_ocr_pdf,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Offline Cache Management")
    
    cache = get_offline_cache()
    cache_size = cache.get_cache_size_mb()
    st.metric("Cached Data Size", f"{cache_size:.2f} MB")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📦 Export Cache"):
            try:
                cache.export_for_offline("offline_package.zip")
                with open("offline_package.zip", "rb") as f:
                    st.download_button(
                        "Download Offline Package",
                        f,
                        "offline_package.zip",
                        "application/zip"
                    )
                st.success("Cache exported successfully!")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with col2:
        if st.button("📥 View Cached Docs"):
            cached = cache.list_cached_documents()
            if cached:
                st.write("**Cached Documents:**")
                for doc in cached:
                    st.write(f"- {doc['name']} ({doc['size_kb']:.1f} KB) - {doc['cached_at']}")
            else:
                st.info("No cached documents yet.")
    
    with col3:
        if st.button("🗑️ Clear Cache", key="clear_cache_btn"):
            if cache.clear_all_cache():
                st.success("Cache cleared!")
                st.rerun()
            else:
                st.error("Failed to clear cache.")
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Actions")
    if st.button("Reset to Default Settings"):
        st.session_state.chunk_size = 500
        st.session_state.chunk_overlap = 50
        st.session_state.top_k = 5
        st.session_state.show_retrieved_context = False
        st.session_state.show_llm_prompt = False
        st.session_state.show_performance_charts = False
        st.session_state.force_ocr_pdf = False
        st.success("Settings reset to defaults.")
    st.markdown("</div>", unsafe_allow_html=True)


def handle_uploads() -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("Upload documents and start querying in natural language.")
    uploaded_files = st.file_uploader(
        "Upload document(s) (PDF, DOCX, TXT, PNG, JPG, JPEG)",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files:
        merged_texts = []
        merged_sections = []
        loaded_names = []
        cache = get_offline_cache()
        
        for file in uploaded_files:
            try:
                parsed = parse_uploaded_file(file, force_ocr=st.session_state.force_ocr_pdf)
                merged_texts.append(parsed.text)
                merged_sections.extend(
                    [(section.text, f"{parsed.name} | {section.source_label}") for section in parsed.sections]
                )
                loaded_names.append(parsed.name)
                
                if parsed.name not in st.session_state.history:
                    st.session_state.history.append(parsed.name)
                
                # Cache document for offline access
                sections_data = [{"text": s.text, "label": s.source_label} for s in parsed.sections]
                if cache.cache_document(parsed.name, parsed.text, sections_data):
                    st.caption(f"✓ {parsed.name} cached for offline access")
                    
            except Exception as exc:
                st.error(f"{file.name}: {exc}")

        if merged_texts:
            st.session_state.doc_text = "\n\n".join(merged_texts)
            st.session_state.doc_sections = merged_sections
            st.session_state.doc_name = loaded_names[0] if len(loaded_names) == 1 else f"{len(loaded_names)} files merged"
            save_history(st.session_state.history)
            st.success(f"Loaded: {', '.join(loaded_names)}")
    return None


def render_search_home() -> None:
    st.markdown('<div class="hero-title">Smart Doc AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI document workspace with grounded answers.</div>', unsafe_allow_html=True)


def render_quick_search() -> None:
    st.markdown(
        '<div class="search-shell-strong"><div class="search-title">Search Workspace</div><div class="muted">Ask from uploaded docs using grounded AI retrieval.</div></div>',
        unsafe_allow_html=True,
    )
    preset_cols = st.columns(4)
    if preset_cols[0].button("Project Objective", use_container_width=True):
        st.session_state.quick_query = "What is the main objective of this document?"
    if preset_cols[1].button("Key Deadlines", use_container_width=True):
        st.session_state.quick_query = "What are important dates or deadlines?"
    if preset_cols[2].button("Important Points", use_container_width=True):
        st.session_state.quick_query = "What are the most important points?"
    if preset_cols[3].button("Summarize Briefly", use_container_width=True):
        st.session_state.quick_query = "Give me a short summary."

    quick_query = st.text_input(
        "Quick Ask",
        key="quick_query",
        placeholder="Ask from uploaded document...",
        label_visibility="collapsed",
    )
    cols = st.columns([1.2, 1.2, 1, 0.9])
    with cols[0]:
        quick_mode = st.selectbox("Mode", ["Focused", "Research", "Writing"], label_visibility="collapsed")
    with cols[1]:
        scope = st.selectbox("Scope", ["Current Document", "All Uploaded"], label_visibility="collapsed")
    with cols[2]:
        st.caption("Grounded Mode")
    with cols[3]:
        quick_go = st.button("Ask")

    if quick_go and quick_query.strip():
        if not st.session_state.doc_text:
            st.warning("Upload a document first.")
            return
        st.session_state.search_history.append(f"Quick Ask: {quick_query}")
        with st.spinner(f"Searching in {quick_mode.lower()} mode..."):
            qa_cfg = get_qa_config()
            quick_result = st.session_state.qa_engine.answer_query(
                quick_query,
                st.session_state.doc_text,
                st.session_state.doc_sections,
                chunk_size=qa_cfg["chunk_size"],
                chunk_overlap=qa_cfg["chunk_overlap"],
                top_k=qa_cfg["top_k"],
            )
        st.session_state.last_result = {"query": quick_query, "result": quick_result}
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Quick Answer")
        render_consistent_text(quick_result.answer)
        st.caption(
            f"Mode: {quick_mode} | Scope: {scope} | Confidence: {quick_result.confidence:.2f} | Source: {quick_result.source_label}"
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_document_tools() -> None:
    if not st.session_state.doc_text:
        st.info("Upload a document to start analysis.")
        return

    top_left, top_right = st.columns([2, 1])
    with top_left:
        st.subheader(st.session_state.doc_name)
        st.caption("Document successfully loaded and indexed.")
    with top_right:
        st.metric("Word count", f"{len(st.session_state.doc_text.split()):,}")

    tab_qa, tab_summary, tab_search, tab_preview, tab_mcq = st.tabs(
        ["Q&A", "Summary", "Keyword Search", "Document Preview", "MCQ Solver"]
    )

    with tab_qa:
        query = st.text_input("Ask a question from the uploaded document", key="qa_query")
        if st.button("Get Answer", type="primary") and query.strip():
            st.session_state.search_history.append(f"Q&A: {query}")
            with st.spinner("Analyzing document..."):
                qa_cfg = get_qa_config()
                result = st.session_state.qa_engine.answer_query(
                    query,
                    st.session_state.doc_text,
                    st.session_state.doc_sections,
                    chunk_size=qa_cfg["chunk_size"],
                    chunk_overlap=qa_cfg["chunk_overlap"],
                    top_k=qa_cfg["top_k"],
                )
            st.session_state.last_result = {"query": query, "result": result}
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_consistent_text(concise_text(result.answer, 280))
            st.caption(f"Confidence: {result.confidence:.2f} | Source: {result.source_label}")
            st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state.show_retrieved_context:
                highlighted = highlight_query_terms(result.relevant_passage, query)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                with st.expander("Show retrieved context"):
                    st.markdown(highlighted, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            if st.session_state.show_llm_prompt:
                st.code(f"Question: {query}\nContext: {result.relevant_passage[:600]}", language="text")
            if st.session_state.show_performance_charts and result.top_matches:
                perf_df = pd.DataFrame(
                    [{"source": source, "score": score} for _, source, score in result.top_matches]
                )
                st.bar_chart(perf_df.set_index("source"))

        if st.session_state.last_result:
            report_query = st.session_state.last_result["query"]
            report_result = st.session_state.last_result["result"]
            report_text = build_report_text(
                st.session_state.doc_name,
                report_query,
                report_result.answer,
                report_result.source_label,
                report_result.confidence,
            )
            st.download_button("Download TXT report", report_text.encode("utf-8"), "document_qa_report.txt")
            pdf_bytes = build_report_pdf_bytes(report_text)
            if pdf_bytes:
                st.download_button("Download PDF report", pdf_bytes, "document_qa_report.pdf", "application/pdf")

    with tab_summary:
        if st.button("Generate summary"):
            with st.spinner("Generating summary..."):
                st.write(st.session_state.qa_engine.summarize(st.session_state.doc_text))

    with tab_search:
        st.markdown("#### Smart Search")
        search_query = st.text_input(
            "Ask a search question",
            placeholder="Example: What is stored procedure?",
            key="smart_search_query",
        )
        if st.button("Search Answer", key="search_answer_btn") and search_query.strip():
            st.session_state.search_history.append(f"Search: {search_query}")
            with st.spinner("Finding best answer..."):
                qa_cfg = get_qa_config()
                s_result = st.session_state.qa_engine.answer_query(
                    search_query,
                    st.session_state.doc_text,
                    st.session_state.doc_sections,
                    chunk_size=qa_cfg["chunk_size"],
                    chunk_overlap=qa_cfg["chunk_overlap"],
                    top_k=qa_cfg["top_k"],
                )
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_consistent_text(concise_text(s_result.answer, 280))
            st.caption(f"Confidence: {s_result.confidence:.2f} | Source: {s_result.source_label}")
            st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state.show_retrieved_context:
                h = highlight_query_terms(s_result.relevant_passage, search_query)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                with st.expander("Show retrieved context"):
                    st.markdown(h, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Keyword Match (optional)")
        keyword = st.text_input("Browse docs by keyword", key="kw")
        if keyword.strip():
            snippets = keyword_snippets(st.session_state.doc_text, keyword)
            if snippets:
                for snippet in snippets:
                    st.write(snippet)
            else:
                st.warning("No match found.")

    with tab_preview:
        preview_chars = st.slider("Preview length", 500, 6000, 2200, 100)
        st.text_area("Extracted text", value=st.session_state.doc_text[:preview_chars], height=350)

    with tab_mcq:
        st.caption("Paste question and options. System picks best option from document context.")
        mcq_question = st.text_input("MCQ Question", key="mcq_question")
        options_text = st.text_area(
            "Options (one per line)",
            placeholder="Option A\nOption B\nOption C\nOption D",
            key="mcq_options",
        )
        if st.button("Solve MCQ"):
            options = [line.strip(" -") for line in options_text.splitlines() if line.strip()]
            with st.spinner("Solving MCQ from document..."):
                result = st.session_state.qa_engine.solve_mcq(
                    question=mcq_question,
                    options=options,
                    document_text=st.session_state.doc_text,
                    sections=st.session_state.doc_sections,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                    top_k=st.session_state.top_k,
                )
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(f"**Selected Option:** {result.selected_option}")
            st.caption(f"Confidence: {result.confidence:.2f}")
            st.write(result.explanation)
            st.markdown("</div>", unsafe_allow_html=True)


init_state()
dark_mode, nav = render_sidebar()
st.markdown(apply_theme(dark_mode), unsafe_allow_html=True)

if not st.session_state.is_logged_in:
    st.markdown('<div class="hero-title">Sign in to continue</div>', unsafe_allow_html=True)
    st.info("Use sidebar Login/Signup to access the workspace.")
else:
    render_search_home()
    handle_uploads()
    if nav == "Search":
        render_quick_search()
        render_document_tools()
    elif nav == "History":
        st.markdown("### Search History")
        if not st.session_state.search_history:
            st.info("No searches yet.")
        for item in reversed(st.session_state.search_history[-100:]):
            st.write(f"- {item}")
    elif nav == "Discover":
        render_about_us()
        render_discover_details()
    elif nav == "More Options":
        st.markdown("## Academics & Education")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Learning Profile")
        subject = st.selectbox(
            "Choose academic subject",
            ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science", "Economics", "English"],
        )
        level = st.selectbox("Choose study level", ["School", "Undergraduate", "Postgraduate", "Competitive Exams"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Study Assistance Options")
        col_a, col_b = st.columns(2)
        with col_a:
            st.checkbox("Generate chapter summary", value=True, key="edu_summary")
            st.checkbox("Extract important formulas", value=False, key="edu_formula")
        with col_b:
            st.checkbox("Create practice questions", value=True, key="edu_questions")
            st.checkbox("Show exam-focused keywords", value=True, key="edu_keywords")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Active Configuration")
        st.write(f"Selected subject: **{subject}**")
        st.write(f"Study level: **{level}**")
        st.write("Upload notes, syllabus, or papers and use Q&A/Keyword search for smart learning.")
        st.markdown("</div>", unsafe_allow_html=True)
        render_document_tools()
    elif nav == "Settings":
        render_settings_panel()
        if st.button("Clear upload history"):
            st.session_state.history = []
            save_history(st.session_state.history)
            st.success("Upload history cleared.")
