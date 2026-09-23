import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from ingest import extract_text_and_split, create_vector_store
from retrieval import load_vector_store, retrieve_relevant_chunks, generate_answer, solve_all_questions
from question_splitter import split_into_questions
from db import db

# Load environment variables from .env file if available (local dev)
load_dotenv()


def get_secret(key: str, default: str = None):
    """
    Retrieves secret primarily from st.secrets, with fallback to os.environ.
    Ensures zero runtime dependence on .env on Streamlit Community Cloud.
    """
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


# Set page configuration with a premium icon and title
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, professional, and modern UI
st.markdown("""
<style>
    /* Gradient header style */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    /* Style for source cards */
    .source-card {
        background-color: #f8f9fa;
        border-left: 5px solid #FF4B4B;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    /* Dark mode adjustments for source cards */
    @media (prefers-color-scheme: dark) {
        .source-card {
            background-color: #1e1e1e;
            border-left: 5px solid #FF4B4B;
        }
    }
    /* Honeypot hidden input styling */
    .honeypot-container, div[data-testid="stTextInput"]:has(input[aria-label="hp_trap_input"]) {
        display: none !important;
        position: absolute !important;
        left: -9999px !important;
        opacity: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- ABUSE PROTECTION: HONEYPOT -----------------
# Invisible honeypot input field: bots auto-fill this, real users never see it.
hp_val = st.text_input("hp_trap_input", key="hp_security_check", label_visibility="collapsed")
if hp_val:
    # Silent reject for automated spam bots
    st.stop()

# ----------------- SESSION IDENTITY SETUP -----------------
# Generate or restore a stable anonymous session ID per visitor (persisted via st.query_params)
if "session_id" not in st.session_state:
    params = st.query_params
    if "session_id" in params and params["session_id"]:
        session_id = str(params["session_id"])
    else:
        session_id = str(uuid.uuid4())
        st.query_params["session_id"] = session_id
    st.session_state.session_id = session_id
else:
    session_id = st.session_state.session_id

# Register session in backend database
db.get_or_create_session(session_id)

# ----------------- SESSION STATE SETUP -----------------
# We use st.session_state and backend DB to persist data across page reruns and browser reloads.
if "current_file" not in st.session_state:
    latest_doc = db.get_latest_document(session_id)
    if latest_doc:
        st.session_state.current_file = latest_doc["filename"]
        st.session_state.current_doc_id = latest_doc["document_id"]
        st.session_state.vector_store = load_vector_store("./chroma_db", session_id=session_id, document_id=latest_doc["document_id"])
    else:
        st.session_state.current_file = None
        st.session_state.current_doc_id = None

if "messages" not in st.session_state:
    cur_doc_id = st.session_state.get("current_doc_id")
    db_msgs = db.get_messages(session_id, document_id=cur_doc_id)
    st.session_state.messages = db.get_messages(session_id, document_id=cur_doc_id)

if "full_document_text" not in st.session_state:
    st.session_state.full_document_text = None

if "solved_results" not in st.session_state:
    cur_doc_id = st.session_state.get("current_doc_id")
    db_solved = db.get_solved_worksheets(session_id, document_id=cur_doc_id)
    st.session_state.solved_results = db_solved if db_solved else None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store("./chroma_db", session_id=session_id, document_id=st.session_state.get("current_doc_id"))

if "target_tab" in st.session_state:
    st.session_state.active_tab = st.session_state.pop("target_tab")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "💬 Chat with Document"

if "detected_questions" not in st.session_state:
    st.session_state.detected_questions = None

# 1. API Key Validation Helper
def validate_groq_key(key: str) -> bool:
    try:
        from groq import Groq
        client = Groq(api_key=key)
        client.models.list()
        return True
    except Exception:
        return False

# ----------------- SETTINGS & ABOUT DIALOG MODAL -----------------
@st.dialog("⚙️ Settings & System")
def show_settings_dialog():
    tab_api, tab_about = st.tabs(["🔑 API & Database", "ℹ️ About & Tech Stack"])
    with tab_api:
        st.subheader("Groq API Configuration")
        active_key = st.session_state.get("user_groq_api_key") or get_secret("GROQ_API_KEY")
        if active_key:
            st.success("✅ Groq API Key is active")
        else:
            st.warning("🔑 Groq API Key required")
            
        entered_key = st.text_input(
            "Enter or Update Groq API Key:",
            type="password",
            value=active_key if active_key else "",
            help="Get a free key from console.groq.com"
        )
        if st.button("Save & Verify API Key", use_container_width=True):
            if entered_key:
                entered_key = entered_key.strip()
                if validate_groq_key(entered_key):
                    os.environ["GROQ_API_KEY"] = entered_key
                    st.session_state["user_groq_api_key"] = entered_key
                    st.success("✅ API key verified and updated!")
                    st.rerun()
                else:
                    st.error("❌ Invalid Groq API key.")

        st.markdown("---")
        st.subheader("Session Data Management")
        st.caption("Clears all ChromaDB vector collections, chat messages, and worksheet results for this session.")
        if st.button("🗑️ Reset Session Vector Database", type="secondary", use_container_width=True):
            try:
                import chromadb
                client = chromadb.PersistentClient(path="./chroma_db")
                for col in client.list_collections():
                    col_name = getattr(col, "name", str(col))
                    if col_name.startswith(f"doc_{session_id}"):
                        client.delete_collection(col_name)
            except Exception:
                pass
            db.clear_session_data(session_id)
            st.session_state.vector_store = None
            st.session_state.current_file = None
            st.session_state.current_doc_id = None
            st.session_state.full_document_text = None
            st.session_state.solved_results = None
            st.session_state.detected_questions = None
            st.session_state.messages = []
            st.session_state.last_uploaded_filename = None
            st.session_state.target_tab = "💬 Chat with Document"
            st.rerun()

    with tab_about:
        st.subheader("System Architecture")
        st.markdown(
            "**Document Q&A Bot** is engineered for high-precision Retrieval-Augmented Generation (RAG) "
            "with multi-document session persistence.\n\n"
            "**Core Tech Stack:**\n"
            "- **Frontend:** Streamlit 1.59 UI with clean collapsible chat architecture\n"
            "- **Document Ingestion:** PyPDF & LangChain RecursiveCharacterTextSplitter\n"
            "- **Vector Database:** ChromaDB with dual session- & document-scoped collections\n"
            "- **Embeddings:** Local HuggingFace `all-MiniLM-L6-v2` (on-device vectorization)\n"
            "- **LLM Engine:** Groq Cloud Llama 3.3 (`openai/gpt-oss-20b`) with deterministic temperature=0.0\n"
            "- **Database Storage:** Supabase (PostgreSQL) with SQLite local fallback"
        )

# ----------------- SIDEBAR CONTENT -----------------
with st.sidebar:
    st.markdown("### 📄 Document Q&A")
    st.caption("Chat with documents & solve worksheets")
    
    # 1. File Uploading
    st.markdown("#### Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

    # Ingestion Logic
    if uploaded_file is not None:
        if st.session_state.get("db_reset_file") == uploaded_file.name:
            st.info("ℹ️ Database reset. Remove file or click below to re-ingest.")
            if st.button("🔄 Re-ingest Document", use_container_width=True):
                st.session_state.db_reset_file = None
                st.session_state.last_uploaded_filename = None
                st.rerun()
        elif st.session_state.get("last_uploaded_filename") != uploaded_file.name:
            if uploaded_file.size > 15 * 1024 * 1024:
                st.error(f"❌ File too large: The uploaded file ({uploaded_file.size / (1024*1024):.1f} MB) exceeds the 15 MB limit.")
            elif not uploaded_file.getvalue().startswith(b"%PDF"):
                st.error("❌ Invalid file format: The uploaded file does not have a valid PDF signature (magic bytes). Please upload a legitimate PDF file.")
            else:
                rate_counts = db.get_rate_limit_counts(session_id, rolling_minutes=60)
                if rate_counts["upload"] >= 5:
                    st.error("⏳ Upload rate limit exceeded: Maximum 5 PDF uploads allowed per hour for this session. Please wait before uploading another document.")
                else:
                    status_placeholder = st.empty()
                    status_placeholder.info(f"New file detected: {uploaded_file.name}")
                    
                    temp_dir = "./temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    with st.spinner("⏳ Parsing PDF, splitting text, and building vector database..."):
                        try:
                            chunks, full_text = extract_text_and_split(temp_file_path)
                            
                            # Store in database to obtain document_id for per-document isolation
                            doc_id = db.save_document(session_id, uploaded_file.name)
                            db.record_rate_limit_action(session_id, "upload")

                            # Create session- and document-isolated vector store
                            vs = create_vector_store(chunks, db_directory="./chroma_db", session_id=session_id, document_id=doc_id)
                            st.session_state.vector_store = vs
                            
                            st.session_state.current_file = uploaded_file.name
                            st.session_state.current_doc_id = doc_id
                            st.session_state.last_uploaded_filename = uploaded_file.name
                            st.session_state.full_document_text = full_text
                            st.session_state.solved_results = None
                            st.session_state.detected_questions = None
                            st.session_state.messages = []
                            st.session_state.target_tab = "💬 Chat with Document"
                            
                            status_placeholder.empty()
                            st.success("✅ Ingestion complete! Ask questions on the right.")
                            st.rerun()
                            
                        except Exception as e:
                            status_placeholder.empty()
                            st.error(f"❌ Error parsing file: {str(e)}")
                        finally:
                            if os.path.exists(temp_file_path):
                                os.remove(temp_file_path)
    
    if st.session_state.get("current_file"):
        st.success(f"📂 Active: {st.session_state.current_file}")

    st.markdown("---")

    # 2. Document History List (Claude-like chat/doc history)
    st.markdown("#### 📚 Document History")
    history_docs = db.get_documents(session_id)
    if history_docs:
        for doc in history_docs:
            is_active = (doc["document_id"] == st.session_state.get("current_doc_id"))
            btn_label = f"{'● ' if is_active else '📄 '}{doc['filename']}"
            if st.button(
                btn_label,
                key=f"hist_doc_{doc['document_id']}",
                type="primary" if is_active else "secondary",
                use_container_width=True
            ):
                if not is_active:
                    st.session_state.current_file = doc["filename"]
                    st.session_state.current_doc_id = doc["document_id"]
                    st.session_state.vector_store = load_vector_store("./chroma_db", session_id=session_id, document_id=doc["document_id"])
                    st.session_state.messages = db.get_messages(session_id, document_id=doc["document_id"])
                    st.session_state.solved_results = db.get_solved_worksheets(session_id, document_id=doc["document_id"])
                    st.session_state.full_document_text = None
                    st.session_state.detected_questions = None
                    st.session_state.target_tab = "💬 Chat with Document"
                    st.rerun()
    else:
        st.caption("No uploaded documents in this session.")

    st.markdown("---")

    # 3. Settings modal trigger button
    if st.button("⚙️ Settings & System", use_container_width=True):
        show_settings_dialog()

# ----------------- MAIN UI CONTENT -----------------
st.markdown('<div class="main-header">Document Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a PDF in the sidebar and ask questions or solve worksheets.</div>', unsafe_allow_html=True)

# 1. Check if a document has been successfully ingested
if not st.session_state.current_file:
    st.info("👈 Please upload and ingest a PDF document in the sidebar to get started.")
else:
    col_tab1, col_tab2 = st.columns([1, 1])
    with col_tab1:
        tab1_active = (st.session_state.active_tab == "💬 Chat with Document")
        if st.button("💬 Chat with Document", type="primary" if tab1_active else "secondary", width="stretch", key="nav_btn_chat"):
            st.session_state.active_tab = "💬 Chat with Document"
            st.rerun()
    with col_tab2:
        tab2_active = (st.session_state.active_tab == "📝 Solve Entire Document")
        if st.button("📝 Solve Entire Document", type="primary" if tab2_active else "secondary", width="stretch", key="nav_btn_solve"):
            st.session_state.active_tab = "📝 Solve Entire Document"
            st.rerun()

    st.markdown("---")
    
    if st.session_state.active_tab == "💬 Chat with Document":
        # 2. Display Chat Messages from History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Display source chunks if they exist for the assistant's message
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("🔍 Show Retrieved Sources"):
                        for idx, doc in enumerate(message["sources"]):
                            page = doc.metadata.get("page", 0) + 1
                            st.markdown(
                                f"<div class='source-card'>"
                                f"<strong>Source {idx+1} (Page {page})</strong><br>"
                                f"{doc.page_content}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

        # 3. Chat Input Box
        if user_query := st.chat_input("Ask a question about the document:"):
            # Check question rate limit (max 20 questions per rolling hour)
            rate_counts = db.get_rate_limit_counts(session_id, rolling_minutes=60)
            if rate_counts["question"] >= 20:
                st.error("⏳ Question rate limit exceeded: Maximum 20 questions allowed per hour for this session. Please wait before asking another question.")
            else:
                # Display the user's message in the chat
                st.session_state.messages.append({"role": "user", "content": user_query})
                db.save_message(session_id, st.session_state.current_doc_id, "user", user_query)
                db.record_rate_limit_action(session_id, "question")
                with st.chat_message("user"):
                    st.markdown(user_query)

                # Generate the response using RAG
                with st.chat_message("assistant"):
                    # Load vector store from session cache or disk
                    vector_store = st.session_state.get("vector_store")
                    if vector_store is None:
                        vector_store = load_vector_store("./chroma_db", session_id=session_id, document_id=st.session_state.get("current_doc_id"))
                        st.session_state.vector_store = vector_store
                    
                    if vector_store is None:
                        error_text = "Error: Vector database could not be loaded. Please re-upload your document."
                        st.markdown(error_text)
                        st.session_state.messages.append({"role": "assistant", "content": error_text})
                        db.save_message(session_id, st.session_state.current_doc_id, "assistant", error_text)
                    else:
                        # Add a spinner while finding context and generating answer
                        with st.spinner("Searching document & generating answer..."):
                            # Step 4: Retrieve relevant chunks
                            retrieved_docs = retrieve_relevant_chunks(user_query, vector_store, k=3)
                            
                            # Step 5: Ask LLM (Groq) with the retrieved context
                            answer, sources = generate_answer(user_query, retrieved_docs)
                            
                            # Display response
                            st.markdown(answer)
                            
                            # Display the sources in an expander
                            if sources:
                                with st.expander("🔍 Show Retrieved Sources"):
                                    for idx, doc in enumerate(sources):
                                        page = doc.metadata.get("page", 0) + 1
                                        st.markdown(
                                            f"<div class='source-card'>"
                                            f"<strong>Source {idx+1} (Page {page})</strong><br>"
                                            f"{doc.page_content}"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
                                        
                            # Store the complete chat round in session state and db
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })
                            db.save_message(session_id, st.session_state.current_doc_id, "assistant", answer)

    else:
        st.subheader("Worksheet Solver Mode")
        st.markdown(
            "This feature parses the worksheet, extracts all numbered questions, and solves each one "
            "individually by retrieving targeted context for every question. This prevents context limits "
            "and rate limit issues, and ensures high accuracy."
        )
        
        # We retrieve the vector store from session cache or load if needed
        vector_store = st.session_state.get("vector_store")
        if vector_store is None:
            vector_store = load_vector_store("./chroma_db", session_id=session_id, document_id=st.session_state.get("current_doc_id"))
            st.session_state.vector_store = vector_store
        
        if vector_store is None:
            st.error("Vector database is not loaded. Please upload a document first.")
        else:
            if st.session_state.solved_results is None:
                if not st.session_state.full_document_text:
                    st.error("Full document text is not available. Please re-ingest your document.")
                else:
                    # Detect questions once and cache in session state for review
                    if st.session_state.detected_questions is None:
                        with st.spinner("Analyzing document structure for questions..."):
                            st.session_state.detected_questions = split_into_questions(st.session_state.full_document_text)
                    
                    questions = st.session_state.detected_questions
                    
                    if not questions:
                        st.warning(
                            "No questions were detected in this document. "
                            "Questions must contain a question mark '?' or start with a question keyword "
                            "(e.g., 'Question 1:', 'Q2.', '1. What is...?')."
                        )
                        # Allow manual entry fallback
                        custom_input = st.text_area("Or enter questions manually to solve (one per line):", key="manual_questions_input")
                        if st.button("➕ Add Manual Questions"):
                            lines = [line.strip() for line in custom_input.strip().split("\n") if line.strip()]
                            if lines:
                                st.session_state.detected_questions = [
                                    {"number": idx + 1, "question_text": l}
                                    for idx, l in enumerate(lines)
                                ]
                                st.rerun()
                            else:
                                st.warning("⚠️ Please enter at least one question before clicking Add.")
                    else:
                        st.info(f"📋 Detected **{len(questions)}** questions. You can review, edit, or remove questions below before solving:")
                        
                        import pandas as pd
                        df_questions = pd.DataFrame([
                            {"Number": q["number"], "Question": q["question_text"]}
                            for q in questions
                        ])
                        
                        edited_df = st.data_editor(
                            df_questions,
                            num_rows="dynamic",
                            width="stretch",
                            column_config={
                                "Number": st.column_config.NumberColumn("Q#", width="small"),
                                "Question": st.column_config.TextColumn("Question Text", width="large", required=True)
                            },
                            key="question_editor"
                        )
                        
                        col_solve, col_redetect = st.columns([3, 1])
                        with col_solve:
                            if st.button("📝 Start Solving Reviewed Questions", width="stretch", type="primary"):
                                reviewed_questions = []
                                for i, row in edited_df.iterrows():
                                    q_text_val = str(row.get("Question", "")).strip()
                                    if q_text_val:
                                        num_val = int(row["Number"]) if pd.notnull(row["Number"]) else i + 1
                                        reviewed_questions.append({"number": num_val, "question_text": q_text_val})
                                
                                if not reviewed_questions:
                                    st.error("No questions in list. Please keep at least one question to solve.")
                                else:
                                    # Create progress bar and status text
                                    progress_bar = st.progress(0.0)
                                    status_text = st.empty()
                                    
                                    def update_progress(current, total):
                                        if total > 0:
                                             percent = min(current / total, 1.0)
                                             progress_bar.progress(percent)
                                             if current < total:
                                                 status_text.markdown(f"⏳ **Solving question {current + 1} of {total}...**")
                                             else:
                                                 status_text.markdown("✅ **Finished solving all questions!**")
                                    
                                    # Solve all questions
                                    results = solve_all_questions(reviewed_questions, vector_store, progress_callback=update_progress)
                                    st.session_state.solved_results = results
                                    db.save_solved_worksheets(session_id, st.session_state.current_doc_id, results)
                                    st.session_state.target_tab = "📝 Solve Entire Document"
                                    st.rerun()
                        with col_redetect:
                            if st.button("🔄 Re-detect Questions", width="stretch"):
                                st.session_state.detected_questions = split_into_questions(st.session_state.full_document_text)
                                st.rerun()
            else:
                # Display results
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.success(f"Successfully solved {len(st.session_state.solved_results)} questions.")
                with col2:
                    if st.button("🔄 Reset Solver", width="stretch"):
                        st.session_state.solved_results = None
                        db.clear_solved_worksheets(session_id)
                        st.session_state.target_tab = "📝 Solve Entire Document"
                        st.rerun()
                
                st.markdown("---")
                for item in st.session_state.solved_results:
                    q_num = item["number"]
                    q_text = item["question_text"]
                    ans = item["answer"]
                    srcs = item["sources"]
                    
                    # Clean title for each question
                    st.markdown(f"### Question {q_num}")
                    st.markdown(q_text)
                    
                    # Highlight answer
                    st.info(ans)
                    
                    # Show sources in expander if available
                    if srcs:
                        with st.expander(f"🔍 Show Sources for Question {q_num}"):
                            for idx, doc in enumerate(srcs):
                                page = doc.metadata.get("page", 0) + 1
                                st.markdown(
                                    f"<div class='source-card'>"
                                    f"<strong>Source {idx+1} (Page {page})</strong><br>"
                                    f"{doc.page_content}"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                    st.markdown("---")
