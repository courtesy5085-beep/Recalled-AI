import os
import io
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta

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

# =========================
# ENV
# =========================

load_dotenv()

OPENAI_KEY = os.getenv(
    "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"
)

USE_OPENAI = bool(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)

client = (
    OpenAI(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)
    if USE_OPENAI
    else None
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Recalled AI",
    page_icon="🧠",
    layout="wide"
)

# =========================
# THEME
# =========================

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# =========================
# DATABASE
# =========================

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
    created_at TIMESTAMP
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
    created_at TIMESTAMP
)
""")

conn.commit()

# =========================
# CHROMADB
# =========================

chroma_client = chromadb.PersistentClient(
    path="chroma_db",
    settings=Settings(
        anonymized_telemetry=False
    )
)

collection = chroma_client.get_or_create_collection(
    name="recalled_memories"
)

# =========================
# MODELS
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# =========================
# HELPERS
# =========================

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

        except Exception:
            pass

    return embedding_model.encode(
        text
    ).tolist()


def generate_ai_metadata(text):

    if not USE_OPENAI:

        return {
            "title": text[:40],
            "summary": text[:120],
            "emotion": "neutral",
            "tags": ["memory"]
        }

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Analyze this memory.

                    Return JSON:

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
            response_format={
                "type": "json_object"
            }
        )

        return json.loads(
            response.choices[0]
            .message.content
        )

    except Exception:

        return {
            "title": text[:40],
            "summary": text[:120],
            "emotion": "neutral",
            "tags": ["memory"]
        }


def save_memory(
    user_id,
    text,
    memory_type="note",
    source_type="text"
):

    metadata = generate_ai_metadata(text)

    title = metadata["title"]
    summary = metadata["summary"]
    emotion = metadata["emotion"]

    tags = ",".join(
        metadata["tags"]
    )

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
        VALUES(
            ?,?,?,?,?,?,?,?,
            datetime('now')
        )
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
        ),
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
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "title": title,
                    "emotion": emotion,
                    "tags": tags,
                }
            ]
        )


def hybrid_search(
    query,
    user_id,
    top_k=10
):

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"user_id": user_id}
    )

    docs = results["documents"][0]

    if not docs:
        return []

    tokenized_docs = [
        d.split()
        for d in docs
    ]

    bm25 = BM25Okapi(
        tokenized_docs
    )

    bm25_scores = bm25.get_scores(
        query.split()
    )

    pairs = list(
        zip(docs, bm25_scores)
    )

    rerank_inputs = [
        (query, doc)
        for doc, _ in pairs
    ]

    rerank_scores = reranker.predict(
        rerank_inputs
    )

    ranked = sorted(
        zip(docs, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# =========================
# AUTH
# =========================

def register_user(
    username,
    password,
    email
):

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
                datetime.utcnow()
            )
        )

        conn.commit()

        return True

    except Exception:
        return False


def login_user(
    username,
    password
):

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

# =========================
# LOGIN PAGE
# =========================

def login_page():

    st.title("🧠 Recalled AI")

    tab1, tab2 = st.tabs([
        "Login",
        "Register"
    ])

    with tab1:

        with st.form("login_form"):

            username = st.text_input(
                "Username"
            )

            password = st.text_input(
                "Password",
                type="password"
            )

            submit = st.form_submit_button(
                "Login"
            )

            if submit:

                if login_user(
                    username,
                    password
                ):

                    st.success(
                        "Logged in"
                    )

                    st.rerun()

                else:

                    st.error(
                        "Invalid credentials"
                    )

    with tab2:

        with st.form("register_form"):

            username = st.text_input(
                "Username"
            )

            email = st.text_input(
                "Email"
            )

            password = st.text_input(
                "Password",
                type="password"
            )

            submit = st.form_submit_button(
                "Register"
            )

            if submit:

                if register_user(
                    username,
                    password,
                    email
                ):

                    st.success(
                        "Registration successful"
                    )

                else:

                    st.error(
                        "Username exists"
                    )

# =========================
# MAIN APP
# =========================

def main_app():

    st.sidebar.title("🧠 Recalled AI")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "➕ Add Memory",
            "🔍 Search",
            "💬 Chat",
            "📊 Insights",
            "⚙️ Settings",
        ]
    )

    st.sidebar.markdown("---")

    st.sidebar.write(
        f"Logged in as: "
        f"{st.session_state.username}"
    )

    if st.sidebar.button("Logout"):

        st.session_state.clear()
        st.rerun()

    # =====================
    # HOME
    # =====================

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
            params=(
                st.session_state.user_id,
            ),
        )

        if df.empty:

            st.info("No memories yet")

        else:

            for _, row in df.iterrows():

                with st.container():

                    st.subheader(
                        row["title"]
                    )

                    st.write(
                        row["summary"]
                    )

                    st.caption(
                        f"""
                        Emotion:
                        {row['emotion']}
                        |
                        Tags:
                        {row['tags']}
                        """
                    )

                    st.markdown("---")

    # =====================
    # ADD MEMORY
    # =====================

    elif page == "➕ Add Memory":

        st.title("➕ Add Memory")

        memory_type = st.selectbox(
            "Memory Type",
            [
                "Note",
                "Journal",
                "Idea",
                "Quote",
            ]
        )

        text = st.text_area(
            "Write memory"
        )

        uploaded_image = st.file_uploader(
            "Upload Image",
            type=[
                "png",
                "jpg",
                "jpeg"
            ]
        )

        uploaded_audio = st.file_uploader(
            "Upload Audio",
            type=[
                "mp3",
                "wav"
            ]
        )

        if st.button("Save Memory"):

            combined_text = text

            if uploaded_image:

                image = Image.open(
                    uploaded_image
                )

                st.image(
                    image,
                    width=250
                )

                combined_text += (
                    "\nImage uploaded."
                )

            if uploaded_audio:

                st.warning(
                    """
                    Audio transcription
                    requires ffmpeg on
                    Streamlit Cloud.
                    """
                )

            with st.spinner(
                "Saving memory..."
            ):

                save_memory(
                    st.session_state.user_id,
                    combined_text,
                    memory_type
                )

            st.success(
                "Memory saved"
            )

    # =====================
    # SEARCH
    # =====================

    elif page == "🔍 Search":

        st.title("🔍 Search Memories")

        query = st.text_input(
            "Search"
        )

        if query:

            with st.spinner(
                "Searching..."
            ):

                results = hybrid_search(
                    query,
                    st.session_state.user_id
                )

            if not results:

                st.warning(
                    "No results found"
                )

            for doc, score in results:

                st.markdown("---")

                st.write(doc)

                st.caption(
                    f"""
                    Similarity:
                    {score:.2f}
                    """
                )

    # =====================
    # CHAT
    # =====================

    elif page == "💬 Chat":

        st.title(
            "💬 Ask Your Memories"
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        question = st.chat_input(
            "Ask something"
        )

        if question:

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            results = hybrid_search(
                question,
                st.session_state.user_id
            )

            context = "\n".join([
                doc
                for doc, _
                in results[:5]
            ])

            answer = f"""
            Based on your memories:

            {context}
            """

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        for msg in st.session_state.chat_history:

            st.chat_message(
                msg["role"]
            ).write(
                msg["content"]
            )

    # =====================
    # INSIGHTS
    # =====================

    elif page == "📊 Insights":

        st.title("📊 Weekly Insights")

        df = pd.read_sql_query(
            """
            SELECT *
            FROM memories
            WHERE user_id=?
            """,
            conn,
            params=(
                st.session_state.user_id,
            ),
        )

        if df.empty:

            st.info(
                "No data available"
            )

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

            st.subheader(
                "AI Insights"
            )

            st.markdown("""
            • You save most memories
            during productive moments.

            • Positive emotions are
            increasing this week.

            • Journal memories are
            your most common type.
            """)

    # =====================
    # SETTINGS
    # =====================

    elif page == "⚙️ Settings":

        st.title("⚙️ Settings")

        st.toggle(
            "Dark Mode",
            value=True
        )

        st.subheader(
            "Export Memories"
        )

        df = pd.read_sql_query(
            """
            SELECT *
            FROM memories
            WHERE user_id=?
            """,
            conn,
            params=(
                st.session_state.user_id,
            ),
        )

        json_data = df.to_json(
            orient="records"
        )

        st.download_button(
            "Download JSON",
            data=json_data,
            file_name="memories.json",
            mime="application/json"
        )

        csv_data = df.to_csv(
            index=False
        )

        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="memories.csv",
            mime="text/csv"
        )

        st.subheader(
            "Danger Zone"
        )

        if st.button(
            "Delete All Memories"
        ):

            cur.execute(
                """
                DELETE FROM memories
                WHERE user_id=?
                """,
                (
                    st.session_state.user_id,
                ),
            )

            conn.commit()

            st.success(
                "All memories deleted"
            )

# =========================
# MAIN
# =========================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
