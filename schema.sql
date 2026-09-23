-- ==============================================================================
-- Supabase (PostgreSQL) Schema for Document Q&A Bot
-- Run this in your Supabase SQL Editor to initialize tables and 30-day cleanup.
-- ==============================================================================

-- 1. Enable pg_cron extension for automated daily retention cleanup
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- 2. Sessions table: tracks anonymous visitor identities and rate limits
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW(),
    upload_count INT DEFAULT 0,
    question_count INT DEFAULT 0
);

-- 3. Documents table: tracks metadata of uploaded PDF documents per session
CREATE TABLE IF NOT EXISTS documents (
    document_id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Messages table: tracks chat conversation history per document/session
CREATE TABLE IF NOT EXISTS messages (
    message_id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    document_id INT REFERENCES documents(document_id) ON DELETE SET NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Solved Worksheets table: tracks batch question solver results per document
CREATE TABLE IF NOT EXISTS solved_worksheets (
    worksheet_id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    document_id INT REFERENCES documents(document_id) ON DELETE SET NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    solved_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Rate limits table: tracks rolling window actions (upload, question)
CREATE TABLE IF NOT EXISTS rate_limits (
    rate_id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    action_type TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- 7. Indices for fast lookup by session_id
CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_solved_worksheets_session ON solved_worksheets(session_id);
CREATE INDEX IF NOT EXISTS idx_rate_limits_session ON rate_limits(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);

-- 7. Daily 30-day retention cleanup scheduled via pg_cron (Runs daily at 03:00 UTC)
-- Note: Cascading foreign keys will automatically delete child messages, documents, and worksheets.
SELECT cron.schedule(
    'daily-session-retention-cleanup',
    '0 3 * * *',
    $$
        DELETE FROM sessions 
        WHERE created_at < NOW() - INTERVAL '30 days';
    $$
);
