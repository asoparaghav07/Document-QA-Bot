import streamlit as st
import os
import uuid
import time
from ingest import extract_text_and_split, create_vector_store
from retrieval import load_vector_store, retrieve_relevant_chunks, generate_answer, solve_all_questions
from question_splitter import split_into_questions
from db import DatabaseManager

st.set_page_config(page_title="Document Q&A Bot", page_icon="🤖", layout="wide")

# Initialize multi-user session identity
if "session_id" in st.query_params:
    session_id = st.query_params["session_id"]
elif "session_id" not in st.session_state:
    session_id = str(uuid.uuid4())
    st.session_state.session_id = session_id
    st.query_params["session_id"] = session_id
else:
    session_id = st.session_state.session_id

db = DatabaseManager()
db.cleanup_expired_sessions(30)

