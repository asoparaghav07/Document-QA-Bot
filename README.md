# Document Q&A Bot using Retrieval-Augmented Generation (RAG)

Welcome to the **Document Q&A Bot**! This is a production-ready, multi-user web application built using **Streamlit**, **LangChain**, **ChromaDB**, **Sentence-Transformers**, and the **Groq API**.

Upload any PDF document, converse with it using natural language, or switch to **Worksheet Solver Mode** to extract and solve questions individually with targeted context and source page citations.

---

## ✨ Features

- 💬 **Interactive Chat with Document**: Query your document in real-time. Answers are grounded strictly in document context with expandable source citations (including page numbers).
- 📝 **Worksheet Solver Mode**:
  - Automatically identifies numbered questions, standard question forms, and imperative prompts (`Name`, `List`, `Describe`, `Explain`, `Identify`, `Define`, `Calculate`, `State`, `Give`, `Compare`).
  - Interactive review table (`st.data_editor`) allowing users to review, edit, add, or delete questions before solving.
  - Solves questions individually with targeted vector retrieval to eliminate context window overflow and LLM rate limits.
  - Clean question extraction without duplicate numbers or trailing answer leakage.
  - Session-state-driven tab persistence across button clicks and Streamlit reruns.
- 👥 **Multi-User Backend & Data Isolation**:
  - Anonymous session identity tracked via `st.query_params["session_id"]`.
  - Scoped vector collections (`doc_{session_id}_{document_id}`) guaranteeing strict per-user and per-document isolation in ChromaDB.
  - Sidebar document history and switching without cross-document chat duplication.
  - Hybrid persistence: Uses **Supabase (PostgreSQL)** when configured, with seamless local **SQLite** fallback for offline development.
  - 30-day data retention cleanup: automatically purges expired database records and cleans stale vector collections.
- 🛡️ **Abuse Prevention & User-Facing Constraints**:
  - **Rate Limiting**: Restricts uploads to **5 uploads/hour** and questions to **20 questions/hour** per rolling session window.
  - **File Size & Page Limits**: Enforces a **15 MB** maximum file size and a **200-page limit** per document.
  - **Magic Bytes Validation**: Verifies `b"%PDF"` file signature to reject renamed non-PDF files immediately.
  - **Bot Protection**: Invisible honeypot form trap stops automated bots (`st.stop()`).

---

## 🛠️ Tech Stack & Architecture

- **Frontend**: Streamlit (with Claude-style clean design, collapsible sidebar, and settings dialog)
- **Document Processing**: PyPDF & LangChain Text Splitters
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (cached in memory with `@st.cache_resource`)
- **Vector Database**: ChromaDB (isolated per session and document via `doc_{session_id}_{document_id}`)
- **LLM Provider**: Groq API (default model: `openai/gpt-oss-20b`, configurable via `GROQ_MODEL` environment variable)
- **Backend Database**: Supabase PostgreSQL (production) / SQLite `storage.db` (local fallback)

---

## 🚦 System Limits & User-Facing Behavior

To ensure fair usage, prevent quota exhaustion, and maintain responsive inference speeds on free-tier infrastructure, the following constraints are enforced:

| Constraint | Limit | Behavior when Exceeded |
| :--- | :--- | :--- |
| **Document Uploads** | **5 uploads per hour** | Surfaces a friendly rate-limit warning showing time until the window resets. |
| **Questions Asked** | **20 questions per hour** | Blocks generation and displays a rate-limit notice. |
| **Maximum File Size** | **15 MB** | Upload is rejected before processing with an error banner. |
| **Maximum Page Count**| **200 pages** | PyPDF raises an informative error before chunking begins. |
| **File Format** | **Valid PDF (`%PDF`)** | Rejects non-PDF files disguised with a `.pdf` extension. |
| **Encrypted Files** | **Unencrypted only** | Surfaces a clear error if a password-protected PDF is uploaded. |

---

## 🗄️ Supabase Multi-User Backend Setup

The application features a hybrid persistence layer:
- **Local Development**: If no Supabase connection string is provided, the app automatically initializes a local SQLite database (`storage.db`). You can run the app immediately without setting up any cloud database.
- **Production / Cloud Deployment**: For shared environments (e.g. Streamlit Community Cloud), configure a **Supabase (PostgreSQL)** project to enable multi-user session tracking, rate limiting, and chat history.

### Required Environment Variables / Secrets

To connect to Supabase, provide the following variables in your `.env` file or Streamlit Cloud Secrets:

| Variable | Description | Example |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | *(Required)* Groq API key for LLM responses | `gsk_...` |
| `GROQ_MODEL` | *(Optional)* Model name (defaults to `openai/gpt-oss-20b`) | `openai/gpt-oss-20b` |
| `SUPABASE_DB_URL` | *(Optional for Cloud)* Direct PostgreSQL connection URI | `postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres` |
| `SUPABASE_URL` | *(Optional)* Supabase Project API URL | `https://[project-ref].supabase.co` |
| `SUPABASE_KEY` | *(Optional)* Supabase anon or service-role API key | `eyJhbGciOi...` |

### Setting Up the Supabase Database

1. Create a free project at [supabase.com](https://supabase.com).
2. Go to **Project Settings** -> **Database** -> **Connection string** (URI mode) to retrieve your connection URI (`SUPABASE_DB_URL`).
3. Open the **SQL Editor** in the Supabase Dashboard, paste the contents of [`schema.sql`](schema.sql), and run the script.
4. The schema creates all necessary tables (`sessions`, `documents`, `messages`, `solved_worksheets`, `rate_limits`) and configures a daily `pg_cron` job to purge data older than 30 days:

```sql
SELECT cron.schedule(
    'delete-expired-sessions-daily',
    '0 3 * * *',
    $$
        DELETE FROM messages WHERE session_id IN (SELECT session_id FROM sessions WHERE created_at < NOW() - INTERVAL '30 days');
        DELETE FROM solved_worksheets WHERE session_id IN (SELECT session_id FROM sessions WHERE created_at < NOW() - INTERVAL '30 days');
        DELETE FROM documents WHERE session_id IN (SELECT session_id FROM sessions WHERE created_at < NOW() - INTERVAL '30 days');
        DELETE FROM rate_limits WHERE session_id IN (SELECT session_id FROM sessions WHERE created_at < NOW() - INTERVAL '30 days');
        DELETE FROM sessions WHERE created_at < NOW() - INTERVAL '30 days';
    $$
);
```

---

## 🚀 Local Setup & Installation

### Prerequisites
- Python 3.9+ installed.
- A free Groq API key from [console.groq.com](https://console.groq.com).

### Step 1: Clone Repository
```bash
git clone https://github.com/asoparaghav07/Document-QA-Bot.git
cd Document-QA-Bot
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Local Environment Variables
Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Populate `.env` with your API keys:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

# Optional: Supabase configuration (omit to use local storage.db SQLite fallback)
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_KEY=your_supabase_anon_or_service_role_key_here
SUPABASE_DB_URL=postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
```

### Step 5: Launch Local Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment to Streamlit Community Cloud

The application is fully prepared for deployment to **Streamlit Community Cloud** with zero runtime dependency on `.env`.

1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **Advanced settings...** -> **Secrets**.
5. Paste your configuration into the Secrets box:

```toml
# Required: Groq API Key
GROQ_API_KEY = "gsk_your_groq_api_key_here"

# Optional: Groq Model (defaults to openai/gpt-oss-20b)
GROQ_MODEL = "openai/gpt-oss-20b"

# Optional: Supabase PostgreSQL URI (if omitted, falls back to local SQLite)
SUPABASE_DB_URL = "postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres"
```

6. Click **Deploy**. Your Document Q&A Bot is live!
