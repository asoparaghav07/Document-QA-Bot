import os
import sqlite3
import datetime
from typing import Optional, List, Dict, Any

try:
    import streamlit as st
except ImportError:
    st = None

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None


def get_db_url() -> Optional[str]:
    """
    Retrieves the Supabase/PostgreSQL connection string from st.secrets or environment.
    Never hardcoded or committed to git.
    """
    if st is not None:
        try:
            if "SUPABASE_DB_URL" in st.secrets:
                return str(st.secrets["SUPABASE_DB_URL"])
            if "DATABASE_URL" in st.secrets:
                return str(st.secrets["DATABASE_URL"])
        except Exception:
            pass
    return os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")


class DatabaseManager:
    """
    Manages persistent storage for Document Q&A Bot.
    Uses Supabase (PostgreSQL) when configured, or local SQLite for offline/testing.
    """

    def __init__(self, db_path: str = "./storage.db"):
        self.db_path = db_path
        self.db_url = get_db_url()
        self.is_postgres = False

        if self.db_url and psycopg2 is not None:
            try:
                conn = psycopg2.connect(self.db_url)
                conn.close()
                self.is_postgres = True
            except Exception as e:
                print(f"Warning: Could not connect to Postgres ({e}), falling back to SQLite.")
                self.is_postgres = False
        
        self.init_schema()

    def get_connection(self):
        if self.is_postgres:
            return psycopg2.connect(self.db_url)
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn

    def init_schema(self):
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            if self.is_postgres:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        last_active TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        uploaded_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        document_id INT,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS solved_worksheets (
                        worksheet_id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        document_id INT,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        solved_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS rate_limits (
                        rate_id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
            else:
                cur.executescript("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        document_id INTEGER,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS solved_worksheets (
                        worksheet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        document_id INTEGER,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        solved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS rate_limits (
                        rate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def get_or_create_session(self, session_id: str):
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            placeholder = "%s" if self.is_postgres else "?"
            cur.execute(f"SELECT session_id FROM sessions WHERE session_id = {placeholder}", (session_id,))
            row = cur.fetchone()
            if not row:
                cur.execute(f"INSERT INTO sessions (session_id) VALUES ({placeholder})", (session_id,))
                conn.commit()
            else:
                cur.execute(
                    f"UPDATE sessions SET last_active = {'NOW()' if self.is_postgres else 'CURRENT_TIMESTAMP'} WHERE session_id = {placeholder}",
                    (session_id,)
                )
                conn.commit()
        finally:
            cur.close()
            conn.close()

    def save_document(self, session_id: str, filename: str) -> int:
        self.get_or_create_session(session_id)
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            p = "%s" if self.is_postgres else "?"
            cur.execute(
                f"SELECT document_id FROM documents WHERE session_id = {p} AND filename = {p} ORDER BY uploaded_at DESC LIMIT 1",
                (session_id, filename)
            )
            existing = cur.fetchone()
            if existing:
                doc_id = existing[0]
                cur.execute(
                    f"UPDATE documents SET uploaded_at = CURRENT_TIMESTAMP WHERE document_id = {p}",
                    (doc_id,)
                )
                conn.commit()
                return doc_id

            if self.is_postgres:
                cur.execute(
                    f"INSERT INTO documents (session_id, filename) VALUES ({p}, {p}) RETURNING document_id",
                    (session_id, filename)
                )
                doc_id = cur.fetchone()[0]
            else:
                cur.execute(
                    f"INSERT INTO documents (session_id, filename) VALUES ({p}, {p})",
                    (session_id, filename)
                )
                doc_id = cur.lastrowid
            conn.commit()
            return doc_id
        finally:
            cur.close()
            conn.close()

    def get_latest_document(self, session_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            p = "%s" if self.is_postgres else "?"
            cur.execute(
                f"SELECT document_id, filename, uploaded_at FROM documents WHERE session_id = {p} ORDER BY uploaded_at DESC LIMIT 1",
                (session_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"document_id": row[0], "filename": row[1], "uploaded_at": row[2]}
        finally:
            cur.close()
            conn.close()

    def get_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all uploaded documents for this session, ordered by upload time descending.
        """
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            p = "%s" if self.is_postgres else "?"
            cur.execute(
                f"SELECT document_id, filename, uploaded_at FROM documents WHERE session_id = {p} ORDER BY uploaded_at DESC",
                (session_id,)
            )
            rows = cur.fetchall()
            return [{"document_id": r[0], "filename": r[1], "uploaded_at": r[2]} for r in rows]
        finally:
            cur.close()
            conn.close()

    def save_message(self, session_id: str, document_id: Optional[int], role: str, content: str):
        self.get_or_create_session(session_id)
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            p = "%s" if self.is_postgres else "?"
            cur.execute(
                f"INSERT INTO messages (session_id, document_id, role, content) VALUES ({p}, {p}, {p}, {p})",
                (session_id, document_id, role, content)
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def get_messages(self, session_id: str, document_id: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            p = "%s" if self.is_postgres else "?"
            if document_id is not None:
                cur.execute(
                    f"SELECT role, content FROM messages WHERE session_id = {p} AND document_id = {p} ORDER BY timestamp ASC",
                    (session_id, document_id)
                )
            else:
                cur.execute(
                    f"SELECT role, content FROM messages WHERE session_id = {p} ORDER BY timestamp ASC",
                    (session_id,)
                )
            rows = cur.fetchall()
            return [{"role": r[0], "content": r[1]} for r in rows]
        finally:
            cur.close()
            conn.close()

    def save_solved_worksheets(self, session_id: str, document_id: Optional[int], results: List[Dict[str, Any]]):
        self.get_or_create_session(session_id)
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            p = "%s" if self.is_postgres else "?"
            for res in results:
                q_text = res.get("question_text", "")
                ans = res.get("answer", "")
                cur.execute(
                    f"INSERT INTO solved_worksheets (session_id, document_id, question, answer) VALUES ({p}, {p}, {p}, {p})",
                    (session_id, document_id, q_text, ans)
                )
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def get_solved_worksheets(self, session_id: str, document_id: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            p = "%s" if self.is_postgres else "?"
            if document_id is not None:
                cur.execute(
                    f"SELECT question, answer FROM solved_worksheets WHERE session_id = {p} AND document_id = {p} ORDER BY solved_at ASC",
                    (session_id, document_id)
                )
            else:
                cur.execute(
                    f"SELECT question, answer FROM solved_worksheets WHERE session_id = {p} ORDER BY solved_at ASC",
                    (session_id,)
                )
            rows = cur.fetchall()
            if not rows:
                return None
            return [{"number": idx + 1, "question_text": r[0], "answer": r[1], "sources": []} for idx, r in enumerate(rows)]
        finally:
            cur.close()
            conn.close()

    def clear_solved_worksheets(self, session_id: str):
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            p = "%s" if self.is_postgres else "?"
            cur.execute(f"DELETE FROM solved_worksheets WHERE session_id = {p}", (session_id,))
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def clear_session_data(self, session_id: str):
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            p = "%s" if self.is_postgres else "?"
            cur.execute(f"DELETE FROM messages WHERE session_id = {p}", (session_id,))
            cur.execute(f"DELETE FROM solved_worksheets WHERE session_id = {p}", (session_id,))
            cur.execute(f"DELETE FROM documents WHERE session_id = {p}", (session_id,))
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def record_rate_limit_action(self, session_id: str, action_type: str):
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            p = "%s" if self.is_postgres else "?"
            cur.execute(
                f"INSERT INTO rate_limits (session_id, action_type) VALUES ({p}, {p})",
                (session_id, action_type)
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def get_rate_limit_counts(self, session_id: str, rolling_minutes: int = 60) -> Dict[str, int]:
        """
        Returns counts of uploads and questions within the rolling time window.
        """
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            p = "%s" if self.is_postgres else "?"
            if self.is_postgres:
                cur.execute(
                    f"""
                    SELECT action_type, COUNT(*) 
                    FROM rate_limits 
                    WHERE session_id = {p} AND timestamp > NOW() - INTERVAL '{rolling_minutes} minutes'
                    GROUP BY action_type
                    """,
                    (session_id,)
                )
            else:
                cur.execute(
                    f"""
                    SELECT action_type, COUNT(*) 
                    FROM rate_limits 
                    WHERE session_id = {p} AND timestamp > datetime('now', '-{rolling_minutes} minutes')
                    GROUP BY action_type
                    """,
                    (session_id,)
                )
            rows = cur.fetchall()
            counts = {"upload": 0, "question": 0}
            for r in rows:
                counts[r[0]] = r[1]
            return counts
        finally:
            cur.close()
            conn.close()

    def cleanup_expired_sessions(self, db_directory: str = "./chroma_db", days: int = 30) -> List[str]:
        """
        Finds sessions older than `days` (default 30 days).
        Deletes their database records AND their associated ChromaDB vector collections
        so disk usage remains bounded.
        """
        import chromadb
        conn = self.get_connection()
        cur = conn.cursor()
        expired_sessions = []
        try:
            if self.is_postgres:
                cur.execute(f"SELECT session_id FROM sessions WHERE created_at < NOW() - INTERVAL '{days} days'")
            else:
                cur.execute(f"SELECT session_id FROM sessions WHERE created_at < datetime('now', '-{days} days')")
            rows = cur.fetchall()
            expired_sessions = [r[0] for r in rows]

            # 1. Delete ChromaDB collections for each expired session
            if expired_sessions and os.path.exists(db_directory):
                try:
                    chroma_client = chromadb.PersistentClient(path=db_directory)
                    existing_cols = [c.name for c in chroma_client.list_collections()]
                    for sid in expired_sessions:
                        col_name = f"doc_{sid}"
                        if col_name in existing_cols:
                            chroma_client.delete_collection(col_name)
                            print(f"Deleted Chroma collection: {col_name}")
                except Exception as e:
                    print(f"Error cleaning Chroma collections: {e}")

            # 2. Delete database rows
            p = "%s" if self.is_postgres else "?"
            for sid in expired_sessions:
                cur.execute(f"DELETE FROM messages WHERE session_id = {p}", (sid,))
                cur.execute(f"DELETE FROM solved_worksheets WHERE session_id = {p}", (sid,))
                cur.execute(f"DELETE FROM documents WHERE session_id = {p}", (sid,))
                cur.execute(f"DELETE FROM rate_limits WHERE session_id = {p}", (sid,))
                cur.execute(f"DELETE FROM sessions WHERE session_id = {p}", (sid,))
            conn.commit()
            return expired_sessions
        finally:
            cur.close()
            conn.close()


# Global database manager instance
db = DatabaseManager()
