import os
import streamlit as st
from dotenv import load_dotenv
from ingest import extract_text_and_split, create_vector_store
from retrieval import load_vector_store, retrieve_relevant_chunks, generate_answer, solve_all_questions
from question_splitter import split_into_questions

# Load environment variables from .env file
load_dotenv()

# Set page configuration with a premium icon and title
st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="🤖",
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
    /* Spinner customization styling */
    .stSpinner > div {
        border-top-color: #FF4B4B !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
# We use st.session_state to persist data across page reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history: list of {"role": "user"/"assistant", "content": "text", "sources": [...]}

if "current_file" not in st.session_state:
    st.session_state.current_file = None  # To track which file is currently in the vector DB

if "full_document_text" not in st.session_state:
    st.session_state.full_document_text = None

if "solved_results" not in st.session_state:
    st.session_state.solved_results = None

# ----------------- SIDEBAR CONTENT -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.title("Settings & Ingestion")
    
    # 1. API Key Configuration
    # We attempt to fetch the Groq key from environment variables.
    # If it is missing, we allow the user to input it directly in the UI.
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.warning("🔑 Groq API Key not found in environment!")
        groq_api_key = st.text_input("Enter Groq API Key:", type="password", help="Get a free key from console.groq.com")
        if groq_api_key:
            # Set the environment variable so retrieval.py can read it
            os.environ["GROQ_API_KEY"] = groq_api_key
            st.success("API key loaded for this session!")
    else:
        st.success("🤖 Groq API Key loaded from .env")

    st.markdown("---")

    # 2. File Uploading
    st.subheader("Upload PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    # 3. Processing and Ingestion Logic
    if uploaded_file is not None:
        # Check if the uploaded file is different from the one currently loaded
        if st.session_state.current_file != uploaded_file.name:
            st.info(f"New file detected: {uploaded_file.name}")
            
            # Ensure a clean directory for storing temporary uploads
            temp_dir = "./temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the uploaded file bytes to a local path so LangChain can load it
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Perform ingestion inside a Streamlit spinner (loading indicator)
            with st.spinner("⏳ Parsing PDF, splitting text, and building vector database..."):
                try:
                    # Step A: Load and Split text
                    chunks, full_text = extract_text_and_split(temp_file_path)
                    
                    # Step B: Create and save vector store
                    create_vector_store(chunks, db_directory="./chroma_db")
                    
                    # Store current file in session state to prevent re-ingesting on every click
                    st.session_state.current_file = uploaded_file.name
                    # Store the full document text for the batch solving feature
                    st.session_state.full_document_text = full_text
                    # Reset solved results for the new file
                    st.session_state.solved_results = None
                    # Clear chat history for the previous document
                    st.session_state.messages = []
                    st.success("✅ Ingestion complete! Ask questions on the right.")
                    
                except Exception as e:
                    st.error(f"❌ Error parsing file: {str(e)}")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
        else:
            st.success(f"📂 Currently Loaded: {uploaded_file.name}")
            
        # Clean Database Button
        if st.button("Reset Vector Database"):
            if os.path.exists("./chroma_db"):
                import shutil
                shutil.rmtree("./chroma_db")
            st.session_state.current_file = None
            st.session_state.full_document_text = None
            st.session_state.solved_results = None
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "**Tech Stack**\n"
        "- Streamlit UI\n"
        "- PyPDF & LangChain\n"
        "- ChromaDB\n"
        "- Sentence-Transformers\n"
        "- Groq Llama 3.3"
    )

# ----------------- MAIN UI CONTENT -----------------
st.markdown('<div class="main-header">Document Q&A Bot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a PDF in the sidebar and ask questions about its content.</div>', unsafe_allow_html=True)

# 1. Check if a document has been successfully ingested
if not st.session_state.current_file:
    st.info("👈 Please upload and ingest a PDF document in the sidebar to get started.")
else:
    # Use tabs to organize Chat and Worksheet Solving
    tab1, tab2 = st.tabs(["💬 Chat with Document", "📝 Solve Entire Document"])
    
    with tab1:
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
            # Display the user's message in the chat
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate the response using RAG
            with st.chat_message("assistant"):
                # Load vector store from disk
                vector_store = load_vector_store("./chroma_db")
                
                if vector_store is None:
                    error_text = "Error: Vector database could not be loaded. Please re-upload your document."
                    st.markdown(error_text)
                    st.session_state.messages.append({"role": "assistant", "content": error_text})
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
                                    
                        # Store the complete chat round in session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

    with tab2:
        st.subheader("Worksheet Solver Mode")
        st.markdown(
            "This feature parses the worksheet, extracts all numbered questions, and solves each one "
            "individually by retrieving targeted context for every question. This prevents context limits "
            "and rate limit issues, and ensures high accuracy."
        )
        
        # We need the vector store to run retrieval for each question
        vector_store = load_vector_store("./chroma_db")
        
        if vector_store is None:
            st.error("Vector database is not loaded. Please upload a document first.")
        else:
            if st.session_state.solved_results is None:
                if st.button("📝 Start Solving Document", use_container_width=True):
                    if not st.session_state.full_document_text:
                        st.error("Full document text is not available. Please re-ingest your document.")
                    else:
                        with st.spinner("Parsing questions..."):
                            questions = split_into_questions(st.session_state.full_document_text)
                        
                        if not questions:
                            st.warning(
                                "No numbered questions were detected in this document. "
                                "Make sure questions start with a number followed by a dot, parenthesis, or colon "
                                "(e.g., '1.', '2)', 'Q3:', 'Question 4:')."
                            )
                        else:
                            st.info(f"Detected {len(questions)} questions. Starting batch solver...")
                            
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
                            results = solve_all_questions(questions, vector_store, progress_callback=update_progress)
                            st.session_state.solved_results = results
                            st.rerun()
            else:
                # Display results
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.success(f"Successfully solved {len(st.session_state.solved_results)} questions.")
                with col2:
                    if st.button("🔄 Reset Solver", use_container_width=True):
                        st.session_state.solved_results = None
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
