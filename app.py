"""
J Street Knowledge Assistant — RAG demo.

Retrieves relevant chunks from the local Chroma vector store built by
ingest.py, then asks Claude to answer using only those chunks, citing which
ones it used. Run with:
    streamlit run app.py

Requires ANTHROPIC_API_KEY, either exported in the shell, set in a local
.env file (loaded automatically below, never committed — see .gitignore),
or set as a Streamlit secret when deployed. Optionally set APP_PASSWORD as
a Streamlit secret to gate the app behind a shared passcode when hosted
publicly — without it, no gate is shown (e.g. local dev).
"""
import chromadb
import streamlit as st
import anthropic
from dotenv import load_dotenv

from ingest import build_index

load_dotenv()

DB_DIR = "chroma_db"
COLLECTION_NAME = "jstreet"
MODEL = "claude-opus-5"
TOP_K = 4

SYSTEM_PROMPT = """You are a research assistant answering questions about J Street \
using only the numbered context chunks provided with each question. Cite the \
chunk number(s) you relied on in brackets, like [1] or [1][3]. If the answer \
isn't in the provided context, say so plainly instead of guessing or using \
outside knowledge."""


def check_password():
    """Gate the app behind a shared passcode if APP_PASSWORD is configured.

    No-op (returns True immediately) when APP_PASSWORD isn't set, so local
    dev is never blocked — only a public deployment needs the gate.
    """
    app_password = st.secrets.get("APP_PASSWORD")
    if not app_password:
        return True

    def password_entered():
        if st.session_state.get("password") == app_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct"):
        return True

    st.text_input("Passcode", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("Incorrect passcode")
    return False


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            return collection
    except Exception:
        pass
    # Cold start (e.g. a fresh hosted deployment) — build the index once.
    collection, _, _ = build_index(client)
    return collection


@st.cache_resource
def get_claude_client():
    return anthropic.Anthropic()


def retrieve(question, collection, k=TOP_K):
    results = collection.query(query_texts=[question], n_results=k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return list(zip(chunks, sources))


def build_context(retrieved):
    blocks = [
        f"[{i}] Source: {source}\n{chunk}"
        for i, (chunk, source) in enumerate(retrieved, start=1)
    ]
    return "\n\n".join(blocks)


st.set_page_config(page_title="J Street Knowledge Assistant", page_icon="🔎")
st.title("J Street Knowledge Assistant")
st.caption(
    "A retrieval-augmented POC that answers questions using J Street's own "
    "public site content — mission, policy positions, FAQ, staff, and more."
)

if not check_password():
    st.stop()

collection = get_collection()
client = get_claude_client()

question = st.text_input("Ask a question about J Street")

if question:
    retrieved = retrieve(question, collection)
    context = build_context(retrieved)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    with st.chat_message("assistant"):
        with client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            st.write_stream(stream.text_stream)

    with st.expander("Retrieved sources"):
        for i, (chunk, source) in enumerate(retrieved, start=1):
            st.markdown(f"**[{i}] {source}**")
            preview = chunk[:500] + ("..." if len(chunk) > 500 else "")
            st.text(preview)
