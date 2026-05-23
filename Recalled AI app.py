import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import hashlib
import os
from datetime import datetime
from PIL import Image
import io
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# ENV
# =========================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("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")
)

USE_OPENAI = bool(os.getenv("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"))

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Recalled Pro",
    page_icon="🧠",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#f5f7fa,#c3cfe2);
}

.memory-card {
    background:white;
    padding:20px;
    border-radius:20px;
    margin-bottom:15px;
    box-shadow:0 4px 8px rgba(0,0,0,0.1);
}

.main-header {
    text-align:center;
    background:linear-gradient(135deg,#667eea,#764ba2);
    color:white;
    padding:20px;
    border-radius:15px;
    margin-bottom:25px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# DATABASE
# =========================
def init_db():
    conn = sqlite3.connect("recalled.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        email TEXT,
        created_at TIMESTAMP
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS memories(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        content TEXT,
        memory_type TEXT,
        tags TEXT,
        summary TEXT,
        emotion TEXT,
        location TEXT,
        created_at TIMESTAMP,
        last_recalled TIMESTAMP,
        recall_count INTEGER DEFAULT 0,
        is_archived BOOLEAN DEFAULT 0,
        embedding TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS user_tags(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        tag_name TEXT,
        count INTEGER DEFAULT 1,
        UNIQUE(user_id, tag_name)
    )
    """)

    conn.commit()
    conn.close()

init_db()

# =========================
# AI FUNCTIONS
# =========================
def summarize_with_ai(text):
    if not USE_OPENAI:
        return text[:120]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role":"user",
                    "content":f"Summarize this in one sentence:\n{text}"
                }
            ],
            max_tokens=60
        )

        return response.choices[0].message.content

    except:
        return text[:120]


def detect_emotion_ai(text):

    emotions = {
        "happy":["love","great","awesome"],
        "sad":["sad","bad","cry"],
        "useful":["guide","tips","learn"],
        "inspiring":["dream","goal","motivation"]
    }

    lower = text.lower()

    for emotion, keywords in emotions.items():
        if any(k in lower for k in keywords):
            return emotion

    return "neutral"


def extract_tags_ai(text):

    words = text.lower().split()

    tags = []

    for word in words:
        if len(word) > 4:
            tags.append(word)

    return list(set(tags[:3]))


def get_embedding(text):

    if not USE_OPENAI:
        return None

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        return response.data[0].embedding

    except:
        return None

# =========================
# AUTH
# =========================
def register_user(username, password, email):

    try:
        conn = sqlite3.connect("recalled.db")
        c = conn.cursor()

        hashed = hashlib.sha256(password.encode()).hexdigest()

        c.execute("""
        INSERT INTO users(username,password,email,created_at)
        VALUES(?,?,?,?)
        """, (username, hashed, email, datetime.now()))

        conn.commit()
        conn.close()

        return True

    except:
        return False


def login_user(username, password):

    conn = sqlite3.connect("recalled.db")
    c = conn.cursor()

    hashed = hashlib.sha256(password.encode()).hexdigest()

    c.execute("""
    SELECT id FROM users
    WHERE username=? AND password=?
    """, (username, hashed))

    user = c.fetchone()

    conn.close()

    if user:
        st.session_state.logged_in = True
        st.session_state.user_id = user[0]
        st.session_state.username = username
        return True

    return False

# =========================
# SAVE MEMORY
# =========================
def save_memory(user_id, title, content, memory_type, location):

    summary = summarize_with_ai(content)

    emotion = detect_emotion_ai(content)

    tags = extract_tags_ai(content)

    embedding = get_embedding(content)

    conn = sqlite3.connect("recalled.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO memories(
        user_id,title,content,memory_type,
        tags,summary,emotion,location,
        created_at,last_recalled,embedding
    )
    VALUES(?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id,
        title,
        content,
        memory_type,
        ",".join(tags),
        summary,
        emotion,
        location,
        datetime.now(),
        datetime.now(),
        json.dumps(embedding) if embedding else None
    ))

    for tag in tags:

        c.execute("""
        INSERT INTO user_tags(user_id,tag_name,count)
        VALUES(?,?,1)
        ON CONFLICT(user_id,tag_name)
        DO UPDATE SET count=count+1
        """, (user_id, tag))

    conn.commit()
    conn.close()

# =========================
# DASHBOARD
# =========================
def dashboard():

    st.markdown(f"""
    <div class="main-header">
    <h1>Welcome {st.session_state.username} 🧠</h1>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("➕ Add Memory")

    with st.form("memory_form"):

        title = st.text_input("Title")

        memory_type = st.selectbox(
            "Type",
            ["Note","Idea","Quote","Article"]
        )

        content = st.text_area("Content")

        location = st.text_input("Location")

        uploaded = st.file_uploader(
            "Upload Image",
            type=["png","jpg","jpeg"]
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, width=200)

        submit = st.form_submit_button("Save")

        if submit:

            if title and content:

                save_memory(
                    st.session_state.user_id,
                    title,
                    content,
                    memory_type,
                    location
                )

                st.success("Memory saved!")

    st.subheader("📚 Your Memories")

    conn = sqlite3.connect("recalled.db")

    df = pd.read_sql_query("""
    SELECT * FROM memories
    WHERE user_id=?
    ORDER BY created_at DESC
    """, conn, params=(st.session_state.user_id,))

    conn.close()

    if not df.empty:

        for _, row in df.iterrows():

            st.markdown(f"""
            <div class="memory-card">
            <h3>{row['title']}</h3>
            <p>{row['summary']}</p>
            <small>Emotion: {row['emotion']}</small>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("No memories yet.")

    if st.sidebar.button("Logout"):

        st.session_state.clear()

        st.rerun()

# =========================
# LOGIN PAGE
# =========================
def login_page():

    st.markdown("""
    <div class="main-header">
    <h1>🧠 Recalled Pro</h1>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login","Register"])

    with tab1:

        with st.form("login"):

            username = st.text_input("Username")

            password = st.text_input(
                "Password",
                type="password"
            )

            submit = st.form_submit_button("Login")

            if submit:

                if login_user(username, password):

                    st.success("Logged in!")

                    st.rerun()

                else:
                    st.error("Invalid credentials")

    with tab2:

        with st.form("register"):

            username = st.text_input("New Username")

            email = st.text_input("Email")

            password = st.text_input(
                "New Password",
                type="password"
            )

            submit = st.form_submit_button("Register")

            if submit:

                if register_user(username, password, email):

                    st.success("Registration successful!")

                else:
                    st.error("Username already exists")

# =========================
# MAIN
# =========================
def main():

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        dashboard()
    else:
        login_page()

if __name__ == "__main__":
    main()
