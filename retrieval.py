import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

def load_vector_store(db_directory: str = "./chroma_db"):
    """
    Loads the persisted Chroma vector database from disk.
    
    Why we do this:
    Instead of recreating the database every time the user asks a question, we load
    the existing database from the disk, saving time and computation.
    """
    if not os.path.exists(db_directory):
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=db_directory,
        embedding_function=embeddings
    )
    return vector_store


def retrieve_relevant_chunks(query: str, vector_store, k: int = 3):
    """
    Step 4: Retrieve the top-k most relevant text chunks from the vector database.
    
    How Similarity Search works:
    1. The query text is converted into an embedding vector using the same embedding model.
    2. ChromaDB calculates the mathematical "distance" (usually cosine similarity) between 
       the query vector and all chunk vectors stored in the database.
    3. The top-k closest vectors (least distance/highest similarity) are returned.
    
    Why k=3?
    To respect Groq's rate limits and keep token usage low, we fetch a small number of chunks.
    This provides enough context to answer most questions without bloating the prompt.
    """
    print(f"Retrieving top {k} relevant chunks for query: '{query}'...")
    # similarity_search returns a list of Document objects containing page_content and metadata
    docs = vector_store.similarity_search(query, k=k)
    return docs


def generate_answer(query: str, retrieved_docs):
    """
    Step 5: Send the retrieved context chunks and the user's question to Groq LLM.
    
    How Generation works:
    1. We format the retrieved chunks into a single "context" block.
    2. We construct a prompt telling the LLM to answer the question using ONLY this context.
    3. We send the prompt to the Groq API and get the answer.
    
    Rate Limit Handling:
    Groq's free tier has limits (30 requests/min, 6000 tokens/min). We use try-except to catch
    errors gracefully if the limits are exceeded, explaining how to resolve it.
    """
    # 1. Check if the Groq API key is set
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "Error: GROQ_API_KEY environment variable is not set. Please add it to your .env file.",
            []
        )
    
    # 2. Format the retrieved document chunks into a single text block
    # We include page numbers so the LLM knows where the information came from (optional but good practice)
    context_list = []
    for i, doc in enumerate(retrieved_docs):
        page_num = doc.metadata.get("page", 0) + 1  # LangChain pages are usually 0-indexed, let's make it 1-indexed
        context_list.append(f"[Source {i+1} - Page {page_num}]:\n{doc.page_content}")
        
    context_text = "\n\n".join(context_list)
    
    # 3. Create a prompt template with fallback to general knowledge.
    # Fallback Logic:
    # If the document context is insufficient or does not contain the answer, the assistant
    # falls back to answering using its own general knowledge. We clearly label these fallback
    # answers with a warning prefix to avoid silently mixing document facts with general knowledge,
    # which could otherwise mislead the user about the reliability and origin of the sources.
    system_instruction = (
        "You are a helpful and precise document Q&A assistant.\n"
        "Your task is to answer the user's question. First, try to answer based strictly on the provided document context.\n"
        "Instructions:\n"
        "1. Attempt to answer using only the facts mentioned in the context. In this case, do NOT add any warning prefix.\n"
        "2. If the context is insufficient or does not contain the answer to the user's question, you must fall back to answering using your own general knowledge.\n"
        "3. CRITICAL: If and only if you answer using your general knowledge, you must start your response with this exact label:\n"
        "⚠️ Answered using general knowledge (not found in document):\n"
        "followed by a space or newline and then your general knowledge answer.\n"
        "4. If you answer using the document context, do NOT show the warning label at all.\n"
        "5. Keep your response clear, concise, and professional.\n\n"
        "--- START CONTEXT ---\n"
        f"{context_text}\n"
        "--- END CONTEXT ---"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{question}")
    ])
    
    # 4. Initialize the Groq Chat model
    # We use llama-3.3-70b-versatile, which is highly capable and fast.
    # We set temperature=0.0 to make the output deterministic and factual (reducing creative hallucination).
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.0
        )
        
        # Format and run the chain
        formatted_messages = prompt.format_messages(question=query)
        response = llm.invoke(formatted_messages)
        return response.content, retrieved_docs
        
    except Exception as e:
        error_msg = str(e).lower()
        # Handle 429 Rate Limit Exceeded
        if "429" in error_msg or "rate_limit_exceeded" in error_msg or "rate limit" in error_msg:
            return (
                "⚠️ **Rate Limit Exceeded (Groq API):** You have exceeded the free tier rate limit "
                "(30 requests/minute or 6,000 tokens/minute). Please wait 10-15 seconds and try asking again.",
                retrieved_docs
            )
        # Handle other API/authentication issues
        elif "authentication" in error_msg or "api key" in error_msg or "unauthorized" in error_msg:
            return (
                "⚠️ **API Key Error:** The provided Groq API key is invalid or unauthorized. "
                "Please verify your `.env` file credentials.",
                []
            )
        else:
            return f"⚠️ **An unexpected error occurred:** {str(e)}", retrieved_docs


def solve_all_questions(questions: list[dict], vector_store, progress_callback=None) -> list[dict]:
    """
    Solves all questions in the provided list.
    For each question, retrieves targeted context chunks and calls generate_answer().
    A 1-second delay is introduced between calls to respect rate limits.
    """
    results = []
    total = len(questions)
    for i, q in enumerate(questions):
        if progress_callback:
            progress_callback(i, total)
            
        q_num = q["number"]
        q_text = q["question_text"]
        
        try:
            # a. Retrieve relevant chunks using the question text
            retrieved_docs = retrieve_relevant_chunks(q_text, vector_store, k=3)
            
            # b. Generate answer using the retrieved context
            answer, sources = generate_answer(q_text, retrieved_docs)
        except Exception as e:
            answer = f"⚠️ **Error generating answer:** {str(e)}"
            sources = []
            
        # c. Store the result
        results.append({
            "number": q_num,
            "question_text": q_text,
            "answer": answer,
            "sources": sources
        })
        
        # d. Introduce a delay to avoid rate limit (30 requests/minute)
        if i + 1 < total:
            time.sleep(1.0)
            
    if progress_callback:
        progress_callback(total, total)
        
    return results
