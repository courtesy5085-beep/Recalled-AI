import os
import json
import sqlite3
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from openai import OpenAI

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from rank_bm25 import BM25Okapi

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# ====================================
# ENV
# ====================================

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

USE_OPENAI = OPENAI_KEY is not None

client = OpenAI(api_key=OPENAI_KEY) if USE_OPENAI else None

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Recalled AI",
    page_icon="🧠",
    layout="wide"
)

# ====================================
# DATABASE
# ====================================

DB_PATH = "recalled.db"

conn = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    email TEXT,
    created_at TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    title TEXT,
    content TEXT,
    summary TEXT,
    emotion TEXT,
    tags TEXT,
    memory_type TEXT,
    source_type TEXT,
    created_at TEXT
)
""")

conn.commit()

# ====================================
# CHROMADB
# ====================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db",
    settings=Settings(
        anonymized_telemetry=False
    )
)

collection = chroma_client.get_or_create_collection(
    name="recalled_memories"
)

# ====================================
# MODELS
# ====================================

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return embedding_model, reranker

embedding_model, reranker = load_models()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# ====================================
# HELPERS
# ====================================

def hash_password(password):

    return hashlib.sha256(
        password.encode()
    ).hexdigest()

def get_embedding(text):

    if USE_OPENAI:

        try:

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )

            return response.data[0].embedding

        except Exception as e:

            st.warning(f"OpenAI embedding failed: {e}")

    return embedding_model.encode(text).tolist()

def generate_ai_metadata(text):

    fallback = {
        "title": text[:40],
        "summary": text[:120],
        "emotion": "neutral",
        "tags": ["memory"]
    }

    if not USE_OPENAI:
        return fallback

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Analyze this memory and return JSON:

{{
"title":"",
"summary":"",
"emotion":"",
"tags":[]
}}

Memory:
{text}
"""
                }
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(
            response.choices[0].message.content
        )

    except Exception as e:

        st.warning(f"Metadata generation failed: {e}")

        return fallback

# ====================================
# MEMORY SAVE
# ====================================

def save_memory(
    user_id,
    text,
    memory_type="note",
    source_type="text"
):

    metadata = generate_ai_metadata(text)

    title = metadata.get("title", "Untitled")
    summary = metadata.get("summary", "")
    emotion = metadata.get("emotion", "neutral")

    tags = ",".join(
        metadata.get("tags", [])
    )

    created_at = datetime.utcnow().isoformat()

    cur.execute(
        """
        INSERT INTO memories(
            user_id,
            title,
            content,
            summary,
            emotion,
            tags,
            memory_type,
            source_type,
            created_at
        )
        VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            title,
            text,
            summary,
            emotion,
            tags,
            memory_type,
            source_type,
            created_at
        )
    )

    conn.commit()

    memory_id = cur.lastrowid

    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):

        embedding = get_embedding(chunk)

        collection.add(
            ids=[f"{memory_id}_{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[
                {
                    "user_id": str(user_id),
                    "memory_id": str(memory_id),
                    "title": title,
                    "emotion": emotion,
                    "tags": tags
                }
            ]
        )

# ====================================
# SEARCH
# ====================================

def hybrid_search(query, user_id, top_k=10):

    try:

        query_embedding = get_embedding(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        docs = results["documents"][0]

        metadatas = results["metadatas"][0]

        filtered_docs = []

        for doc, meta in zip(docs, metadatas):

            if meta.get("user_id") == str(user_id):
                filtered_docs.append(doc)

        if not filtered_docs:
            return []

        tokenized_docs = [
            d.split()
            for d in filtered_docs
        ]

        bm25 = BM25Okapi(tokenized_docs)

        bm25_scores = bm25.get_scores(
            query.split()
        )

        rerank_inputs = [
            (query, doc)
            for doc in filtered_docs
        ]

        rerank_scores = reranker.predict(
            rerank_inputs
        )

        ranked = sorted(
            zip(filtered_docs, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked

    except Exception as e:

        st.error(f"Search failed: {e}")

        return []

# ====================================
# AUTH
# ====================================

def register_user(username, password, email):

    try:

        cur.execute(
            """
            INSERT INTO users(
                username,
                password,
                email,
                created_at
            )
            VALUES(?,?,?,?)
            """,
            (
                username,
                hash_password(password),
                email,
                datetime.utcnow().isoformat()
            )
        )

        conn.commit()

        return True

    except Exception:
        return False

def login_user(username, password):

    cur.execute(
        """
        SELECT id
        FROM users
        WHERE username=?
        AND password=?
        """,
        (
            username,
            hash_password(password)
        )
    )

    user = cur.fetchone()

    if user:

        st.session_state.logged_in = True
        st.session_state.user_id = user[0]
        st.session_state.username = username

        return True

    return False

# ====================================
# LOGIN PAGE
# ====================================

def login_page():

    st.title("🧠 Recalled AI")

    tab1, tab2 = st.tabs([
        "Login",
        "Register"
    ])

    with tab1:

        with st.form("login_form"):

            username = st.text_input("Username")

            password = st.text_input(
                "Password",
                type="password"
            )

            submit = st.form_submit_button("Login")

            if submit:

                if login_user(username, password):

                    st.success("Logged in")

                    st.rerun()

                else:

                    st.error("Invalid credentials")

    with tab2:

        with st.form("register_form"):

            username = st.text_input("New Username")

            email = st.text_input("Email")

            password = st.text_input(
                "New Password",
                type="password"
            )

            submit = st.form_submit_button("Register")

            if submit:

                if register_user(
                    username,
                    password,
                    email
                ):

                    st.success("Registration successful")

                else:

                    st.error("Username already exists")

# ====================================
# MAIN APP
# ====================================

def main_app():

    st.sidebar.title("🧠 Recalled AI")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "➕ Add Memory",
            "🔍 Search",
            "📊 Insights",
            "⚙️ Settings"
        ]
    )

    st.sidebar.write(
        f"Logged in as: {st.session_state.username}"
    )

    if st.sidebar.button("Logout"):

        st.session_state.clear()
        st.rerun()

    # HOME

    if page == "🏠 Home":

        st.title("🏠 Memory Timeline")

        df = pd.read_sql_query(
            """
            SELECT *
            FROM memories
            WHERE user_id=?
            ORDER BY created_at DESC
            """,
            conn,
            params=(st.session_state.user_id,)
        )

        if df.empty:

            st.info("No memories yet")

        else:

            for _, row in df.iterrows():

                st.subheader(row["title"])

                st.write(row["summary"])

                st.caption(
                    f"Emotion: {row['emotion']} | Tags: {row['tags']}"
                )

                st.markdown("---")

    # ADD MEMORY

    elif page == "➕ Add Memory":

        st.title("➕ Add Memory")

        memory_type = st.selectbox(
            "Memory Type",
            [
                "Note",
                "Journal",
                "Idea",
                "Quote"
            ]
        )

        text = st.text_area("Write memory")

        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg"]
        )

        if st.button("Save Memory"):

            combined_text = text

            if uploaded_image:

                image = Image.open(uploaded_image)

                st.image(image, width=250)

                combined_text += "\nImage uploaded."

            if combined_text.strip() == "":

                st.warning("Please enter memory")

            else:

                with st.spinner("Saving memory..."):

                    save_memory(
                        st.session_state.user_id,
                        combined_text,
                        memory_type
                    )

                st.success("Memory saved")

    # SEARCH

    elif page == "🔍 Search":

        st.title("🔍 Search Memories")

        query = st.text_input("Search")

        if query:

            results = hybrid_search(
                query,
                st.session_state.user_id
            )

            if not results:

                st.warning("No results found")

            for doc, score in results:

                st.markdown("---")

                st.write(doc)

                st.caption(
                    f"Similarity: {score:.2f}"
                )

    # INSIGHTS

    elif page == "📊 Insights":

        st.title("📊 Insights")

        df = pd.read_sql_query(
            """
            SELECT *
            FROM memories
            WHERE user_id=?
            """,
            conn,
            params=(st.session_state.user_id,)
        )

        if df.empty:

            st.info("No data available")

        else:

            emotion_counts = (
                df["emotion"]
                .value_counts()
                .reset_index()
            )

            emotion_counts.columns = [
                "Emotion",
                "Count"
            ]

            fig = px.bar(
                emotion_counts,
                x="Emotion",
                y="Count",
                title="Mood Trends"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # SETTINGS

    elif page == "⚙️ Settings":

        st.title("⚙️ Settings")

        df = pd.read_sql_query(
            """
            SELECT *
            FROM memories
            WHERE user_id=?
            """,
            conn,
            params=(st.session_state.user_id,)
        )

        st.subheader("Export Memories")

        st.download_button(
            "Download JSON",
            data=df.to_json(orient="records"),
            file_name="memories.json",
            mime="application/json"
        )

        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name="memories.csv",
            mime="text/csv"
        )

# ====================================
# MAIN
# ====================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
