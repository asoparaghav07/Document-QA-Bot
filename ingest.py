import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def extract_text_and_split(pdf_file_path: str):
    """
    Step 1 & 2: Load the PDF document and split its content into smaller text chunks.
    
    Why Chunking is important:
    LLMs (like Groq's Llama model) have a context window limit (maximum tokens they can read at once).
    Additionally, passing a massive document to an LLM is slow and expensive. 
    By chunking the document into small pieces (e.g., 1000 characters), we can retrieve 
    and send only the most relevant snippets to the LLM.
    
    Why Overlap is important:
    When splitting text, sentences can get cut in half at the boundary. An overlap (e.g., 200 characters)
    ensures that context is preserved between consecutive chunks so no information is lost at the cuts.
    """
    print(f"Loading PDF from: {pdf_file_path}...")
    loader = PyPDFLoader(pdf_file_path)
    
    # Load and parse pages from the PDF file.
    # PyPDFLoader parses the PDF and creates a list of Document objects,
    # each containing page content and metadata (like page number).
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} pages.")

    # We use RecursiveCharacterTextSplitter because it splits text by trying different
    # separators in order (like double newlines, single newlines, spaces) to keep 
    # paragraphs and sentences intact as much as possible.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Maximum size of each text chunk (in characters)
        chunk_overlap=200,     # Overlap between consecutive chunks (in characters)
        length_function=len    # Use Python's len() to measure characters
    )
    
    print("Splitting document into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Split pages into {len(chunks)} text chunks.")
    
    # Reconstruct the full original document text (before chunking)
    full_text = "\n".join([doc.page_content for doc in documents])
    
    return chunks, full_text


def create_vector_store(chunks, db_directory: str = "./chroma_db"):
    """
    Step 3: Convert text chunks into vector embeddings and store them in a local ChromaDB database.
    
    Why Embeddings?
    Computers cannot understand the raw meaning of words, but they can understand numbers. 
    An embedding model converts a text chunk into a high-dimensional vector (a list of numbers) 
    representing its semantic meaning. If two text chunks talk about similar topics, 
    their vectors will be close together in space.
    
    Why the Embedding Model ('all-MiniLM-L6-v2')?
    We use Hugging Face's 'all-MiniLM-L6-v2' model because it runs entirely locally on your machine 
    for free. It is fast, lightweight, and very effective for semantic search tasks.
    
    Why a Vector Database (ChromaDB)?
    Standard databases query text by exact keyword matches. Vector databases store text along with 
    their numerical embeddings. This allows us to perform "Similarity Search", finding chunks 
    that mean something similar to a user's question, even if they use completely different words!
    """
    # 1. Initialize the embedding model. This will download the model (~90MB) on first run
    # and run it locally on your CPU/GPU for subsequent runs.
    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Clean up any existing vector store directory if we are doing a fresh ingestion
    if os.path.exists(db_directory):
        print(f"Clearing existing vector database at {db_directory}...")
        # We use shutil.rmtree to fully remove the folder so Chroma starts clean
        shutil.rmtree(db_directory)
        
    # 3. Create the Chroma database from our document chunks
    print(f"Creating vector database at '{db_directory}'...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_directory
    )
    
    # In older LangChain versions, .persist() was required to save to disk.
    # In newer versions of Chroma, saving is done automatically.
    print("Vector database successfully built and saved to disk.")
    return vector_store
