# Document Q&A Bot using Retrieval-Augmented Generation (RAG)

Welcome to the **Document Q&A Bot** project! This is a simple, modular, and beginner-friendly Python web application built using **Streamlit**, **LangChain**, **ChromaDB**, and the **Groq API**. 

With this application, you can upload any PDF document, ask natural language questions, and receive accurate answers derived strictly from the text in your document, complete with source page references to verify correctness and prevent AI hallucinations.

---

## 📖 Educational Guide: How RAG Works

As a beginner, this project is a great way to understand the core pillars of modern Generative AI development. Here are the explanations for the concepts you need to know:

### 1. What is RAG and why is it useful?
**RAG (Retrieval-Augmented Generation)** is a design pattern used to solve two major limitations of Large Language Models (LLMs):
* **Lack of Private/Real-time Knowledge**: LLMs are frozen in time after training and cannot see your personal files or internal company documents.
* **Hallucinations**: When asked about things they don't know, LLMs often make up plausible-sounding but completely false answers.

RAG fixes this by **retrieving** relevant snippets from your document first, pasting them into a prompt as **context**, and then asking the LLM to **generate** an answer based *only* on that context. The LLM acts as an "open-book reader" rather than answering from memory.

### 2. How do Chunking and Embeddings work?
* **Chunking**: An LLM cannot process a 200-page book all at once due to strict limit sizes (context windows). To solve this, we chop the document into small pieces (chunks) of a fixed size (e.g., 1,000 characters). We include a small "overlap" (e.g., 200 characters) between chunks to ensure context (like sentences split at boundaries) is preserved.
* **Embeddings**: An embedding is a way of translating text into numbers that a computer can understand. An embedding model (like `all-MiniLM-L6-v2`) converts a chunk of text into a list of numbers (a high-dimensional vector). Text chunks with similar meanings will have vectors close to each other in mathematical space, enabling **semantic search** (searching by meaning, not just exact keywords).

### 3. Why use a Vector Database instead of sending the whole document to the LLM?
* **Token Limits & Cost**: Sending a huge document on every query quickly exceeds the LLM's input limit and is extremely slow/costly.
* **Speed**: A Vector Database (like **ChromaDB**) stores the text chunks along with their numerical embeddings. It is optimized to search millions of vectors in milliseconds to find the top-3 or top-4 chunks closest to your question. We then send *only* those few relevant chunks to the LLM, keeping it fast and cheap.

### 4. What is the Retrieval Step actually doing?
When you submit a question:
1. The app converts your question into a mathematical vector using the same embedding model.
2. It queries ChromaDB to find the vectors that are closest to your question's vector (using similarity metrics).
3. ChromaDB returns the corresponding text chunks (and metadata, like page numbers).
4. These chunks are stuffed into a prompt template alongside your question, instructing the LLM: *"Read this context and answer the question. If it's not here, say 'I don't know'."*

---

## 🛠️ Tech Stack & Architecture

* **Frontend**: `Streamlit` (for a clean, fast web UI)
* **Orchestration**: `LangChain` (for loader, splitter, embeddings interface, and model chains)
* **Local Vector Storage**: `ChromaDB`
* **Local Embedding Model**: `sentence-transformers` (`all-MiniLM-L6-v2`, runs locally for free!)
* **LLM Generation**: `Groq API` running the `llama-3.3-70b-versatile` model (high-quality, fast, free tier)

---

## 🚀 Setup & Installation Instructions

Follow these steps to get the project running locally:

### Prerequisites
Make sure you have **Python 3.9+** installed on your system.

### Step 1: Clone or Navigate to the Directory
Go into the project folder:
```bash
cd "/home/raghav/Desktop/Document Q&A Bot"
```

### Step 2: Set up a Virtual Environment (Recommended)
Create a Python virtual environment to isolate the project dependencies:
```bash
python3 -m venv venv
```
Activate the virtual environment:
* **On Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```
* **On Windows:**
  ```bash
  venv\Scripts\activate
  ```

### Step 3: Install Dependencies
Install all the required Python libraries listed in `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Note: Installing `sentence-transformers` might take a moment as it sets up PyTorch and local models.*

### Step 4: Get a Free Groq API Key
1. Go to [console.groq.com](https://console.groq.com/) and log in (or create a free account).
2. Go to the **API Keys** section in the sidebar.
3. Click **Create API Key**, name it (e.g., `Document Bot`), and copy the key.

### Step 5: Configure Environment Variables
1. Copy the template `.env.example` file and rename it to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open the `.env` file in a text editor and replace `your_groq_api_key_here` with your actual Groq API key:
   ```env
   GROQ_API_KEY=gsk_yOuRaCtUaLkEyHeRe...
   ```

---

## 🏃 How to Run the App

Start the Streamlit development server:
```bash
streamlit run app.py
```

Once running, Streamlit will open a browser window automatically (usually at `http://localhost:8501`).

### How to use it:
1. **Upload a PDF** in the sidebar. You will see a spinner indicating the text is being processed and stored in ChromaDB.
2. Once complete, you will see a success message.
3. **Ask questions** in the chat box on the right.
4. Expand the **"Show Retrieved Sources"** accordion under the response to see which exact pages and text chunks the model used to compose the answer!
