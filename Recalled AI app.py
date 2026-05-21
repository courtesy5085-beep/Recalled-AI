import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import hashlib
import os
from datetime import datetime, timedelta
from PIL import Image
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import re
from dotenv import load_dotenv
import openai
import speech_recognition as sr
from pydub import AudioSegment
import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables (API keys here, no .toml)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# If you don't have OpenAI key, the app falls back to rule‑based mode
USE_OPENAI = bool(openai.api_key)

# ---------- Page Config ----------
st.set_page_config(
    page_title="Recalled Pro – AI Memory Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS (same as before, enhanced) ----------
st.markdown("""
<style>
    /* Same as earlier but I keep it concise for answer */
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .memory-card { background: white; border-radius: 20px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.3s; }
    .memory-card:hover { transform: translateY(-5px); }
    .main-header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin-bottom: 30px; }
    .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 20px; color: white; text-align: center; margin: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------- Database ----------
def init_db():
    conn = sqlite3.connect('recalled.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, email TEXT, created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS memories
                 (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, content TEXT, memory_type TEXT,
                  tags TEXT, summary TEXT, emotion TEXT, location TEXT, created_at TIMESTAMP,
                  last_recalled TIMESTAMP, recall_count INTEGER DEFAULT 0, is_archived BOOLEAN DEFAULT 0,
                  embedding BLOB, FOREIGN KEY(user_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS recall_events
                 (id INTEGER PRIMARY KEY, memory_id INTEGER, user_id INTEGER, recalled_at TIMESTAMP, feedback TEXT)''')
    conn.commit()
    conn.close()
init_db()

# ---------- AI Functions with OpenAI ----------
def summarize_with_ai(text: str) -> str:
    if USE_OPENAI:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Summarize this in one sentence:\n{text}"}],
                max_tokens=60
            )
            return response.choices[0].message.content.strip()
        except:
            return text[:150] + "..."
    else:
        return text[:150] + "..." if len(text) > 150 else text

def detect_emotion_ai(text: str) -> str:
    if USE_OPENAI:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Classify the emotion (happy, sad, exciting, inspiring, useful, neutral) of:\n{text}"}],
                max_tokens=10
            )
            emotion = response.choices[0].message.content.strip().lower()
            if emotion in ["happy","sad","exciting","inspiring","useful"]:
                return emotion
        except:
            pass
    # fallback
    emotions = {'happy':['love','great'], 'sad':['sad','sorry'], 'exciting':['exciting','awesome'], 'inspiring':['inspire','dream'], 'useful':['tip','guide']}
    text_low = text.lower()
    for em, kw in emotions.items():
        if any(k in text_low for k in kw):
            return em
    return "neutral"

def extract_tags_ai(text: str) -> List[str]:
    if USE_OPENAI:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Extract up to 3 short tags (comma separated) from:\n{text}"}],
                max_tokens=20
            )
            tags = [t.strip().lower() for t in response.choices[0].message.content.split(",")]
            return tags[:3]
        except:
            pass
    # fallback
    topics = {'tech':['ai','code'], 'health':['fitness','diet'], 'travel':['trip','hotel'], 'food':['recipe','cooking']}
    tags = set()
    for topic, kw in topics.items():
        if any(k in text.lower() for k in kw):
            tags.add(topic)
    return list(tags) or ["general"]

def get_embedding(text: str):
    if USE_OPENAI:
        try:
            response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
            return response['data'][0]['embedding']
        except:
            return None
    return None

# ---------- Memory Management with Embeddings ----------
def save_memory_advanced(user_id, title, content, memory_type, location=""):
    summary = summarize_with_ai(content)
    emotion = detect_emotion_ai(content)
    tags = extract_tags_ai(content)
    tags_str = ",".join(tags)
    embedding = get_embedding(content)
    emb_blob = json.dumps(embedding) if embedding else None

    conn = sqlite3.connect('recalled.db')
    c = conn.cursor()
    c.execute('''INSERT INTO memories 
                 (user_id, title, content, memory_type, tags, summary, emotion, location, created_at, last_recalled, embedding)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
              (user_id, title, content, memory_type, tags_str, summary, emotion, location,
               datetime.now(), datetime.now(), emb_blob))
    memory_id = c.lastrowid
    # Update user_tags
    for tag in tags:
        c.execute('''INSERT INTO user_tags (user_id, tag_name, count) VALUES (?,?,1)
                     ON CONFLICT(user_id, tag_name) DO UPDATE SET count = count+1''', (user_id, tag))
    conn.commit()
    conn.close()
    return memory_id

def smart_recall_ranking(user_id, limit=5):
    """Score forgotten memories using AI + recency decay"""
    conn = sqlite3.connect('recalled.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''SELECT *, julianday('now') - julianday(last_recalled) as days_since_recall,
                        julianday('now') - julianday(created_at) as age_days
                 FROM memories WHERE user_id=? AND is_archived=0''', (user_id,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return []
    # Score = (days_since_recall * 0.6) + ( (30 - recall_count) * 0.4 ) + (age_days * 0.2)
    scored = []
    for r in rows:
        days_since = r['days_since_recall'] or 0
        recall_penalty = max(0, 20 - (r['recall_count'] or 0))
        age = min(r['age_days'] or 0, 60)
        score = days_since * 0.6 + recall_penalty * 0.3 + age * 0.1
        scored.append((score, dict(r)))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [item[1] for item in scored[:limit]]

# ---------- Similarity Search using embeddings ----------
def find_similar_memories(user_id, query_text, top_k=3):
    query_emb = get_embedding(query_text)
    if not query_emb:
        return []
    conn = sqlite3.connect('recalled.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id, title, content, embedding FROM memories WHERE user_id=? AND embedding IS NOT NULL", (user_id,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return []
    # Compute cosine similarity manually (simplified)
    similarities = []
    for r in rows:
        emb = json.loads(r['embedding'])
        if emb:
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            similarities.append((sim, r['id'], r['title'], r['content']))
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_k]

# ---------- Authentication (unchanged) ----------
def login_user(username, password):
    conn = sqlite3.connect('recalled.db')
    c = conn.cursor()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, hashed))
    user = c.fetchone()
    conn.close()
    if user:
        st.session_state['user_id'] = user[0]
        st.session_state['username'] = username
        st.session_state['logged_in'] = True
        return True
    return False

def register_user(username, password, email):
    try:
        conn = sqlite3.connect('recalled.db')
        c = conn.cursor()
        hashed = hashlib.sha256(password.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password, email, created_at) VALUES (?,?,?,?)",
                  (username, hashed, email, datetime.now()))
        conn.commit()
        conn.close()
        return True
    except:
        return False

# ---------- UI Components ----------
def login_page():
    st.markdown('<div class="main-header"><h1>🧠 Recalled Pro</h1><p>AI‑Powered Memory Companion</p></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            with st.form("login"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    if login_user(user, pwd):
                        st.success("Welcome back!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        with tab2:
            with st.form("register"):
                new_user = st.text_input("Username")
                email = st.text_input("Email")
                pwd1 = st.text_input("Password", type="password")
                pwd2 = st.text_input("Confirm", type="password")
                if st.form_submit_button("Register"):
                    if pwd1 != pwd2:
                        st.error("Passwords mismatch")
                    elif len(pwd1) < 6:
                        st.error("Password too short")
                    elif register_user(new_user, pwd1, email):
                        st.success("Registered! Login now.")
                    else:
                        st.error("Username exists")

def dashboard():
    st.markdown(f'<div class="main-header"><h1>Welcome, {st.session_state["username"]} 🧠</h1></div>', unsafe_allow_html=True)
    
    # Stats
    conn = sqlite3.connect('recalled.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM memories WHERE user_id=?", (st.session_state['user_id'],))
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT tags) FROM memories WHERE user_id=?", (st.session_state['user_id'],))
    tag_count = c.fetchone()[0] or 0
    c.execute("SELECT AVG(recall_count) FROM memories WHERE user_id=?", (st.session_state['user_id'],))
    avg_recall = c.fetchone()[0] or 0
    conn.close()
    
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total Memories", total)
    col2.metric("Unique Tags", tag_count)
    col3.metric("Avg Recalls", f"{avg_recall:.1f}")
    col4.metric("AI Status", "ON" if USE_OPENAI else "Fallback")
    
    # Smart Recalls
    st.subheader("🔮 Smart Recalls (AI‑ranked)")
    recalls = smart_recall_ranking(st.session_state['user_id'], limit=3)
    if recalls:
        for mem in recalls:
            with st.container():
                st.markdown(f"""
                <div class="memory-card">
                    <h3>📌 {mem['title']}</h3>
                    <p><em>{mem['summary']}</em></p>
                    <p>🎭 {mem['emotion']} | 🏷️ {mem['tags']} | 🔄 {mem['recall_count']} recalls</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Recall this", key=f"rec_{mem['id']}"):
                    # Update recall count
                    conn = sqlite3.connect('recalled.db')
                    c = conn.cursor()
                    c.execute("UPDATE memories SET recall_count=recall_count+1, last_recalled=? WHERE id=?", (datetime.now(), mem['id']))
                    conn.commit()
                    conn.close()
                    st.success("Recalled! Memory strengthened. ✨")
                    st.rerun()
    else:
        st.info("No memories yet. Add your first one below.")
    
    # Add Memory
    with st.expander("➕ Add New Memory", expanded=False):
        with st.form("new_memory"):
            title = st.text_input("Title")
            mem_type = st.selectbox("Type", ["Note","Idea","Article","Quote","Other"])
            content = st.text_area("Content", height=150)
            location = st.text_input("Location (optional)")
            uploaded_file = st.file_uploader("Or upload image/audio", type=["png","jpg","jpeg","wav","mp3"])
            if uploaded_file:
                if uploaded_file.type.startswith("image"):
                    img = Image.open(uploaded_file)
                    st.image(img, width=200)
                    # AI image caption (openai)
                    if USE_OPENAI:
                        import base64
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG")
                        img_b64 = base64.b64encode(buffered.getvalue()).decode()
                        response = openai.ChatCompletion.create(
                            model="gpt-4-vision-preview",
                            messages=[{"role":"user","content":[
                                {"type":"text","text":"Describe this image in one sentence."},
                                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}}
                            ]}],
                            max_tokens=100
                        )
                        content = response.choices[0].message.content
                        st.info(f"AI vision: {content}")
                elif uploaded_file.type.startswith("audio"):
                    st.audio(uploaded_file)
                    # Transcribe with speech_recognition
                    with open("temp_audio.wav","wb") as f:
                        f.write(uploaded_file.getbuffer())
                    r = sr.Recognizer()
                    with sr.AudioFile("temp_audio.wav") as source:
                        audio = r.record(source)
                    try:
                        content = r.recognize_google(audio)
                        st.success(f"Transcribed: {content}")
                    except:
                        st.warning("Could not transcribe audio.")
            submit = st.form_submit_button("Save Memory")
            if submit and title and content:
                save_memory_advanced(st.session_state['user_id'], title, content, mem_type, location)
                st.success("Memory saved with AI magic! ✨")
                st.rerun()
    
    # Search & Similarity
    st.subheader("🔎 Find Similar Memories")
    query = st.text_input("Describe what you're looking for (e.g., 'travel ideas')")
    if query and st.button("Search"):
        similar = find_similar_memories(st.session_state['user_id'], query, top_k=3)
        if similar:
            for score, mem_id, title, content_snippet in similar:
                st.markdown(f"**{title}** (score: {score:.2f})<br><small>{content_snippet[:150]}...</small>", unsafe_allow_html=True)
        else:
            st.info("No similar memories found.")
    
    # Logout
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100?text=Recalled+Pro", use_column_width=True)
        st.markdown(f"**{st.session_state['username']}**")
        if st.button("🚪 Logout"):
            for key in ['user_id','username','logged_in']:
                st.session_state.pop(key, None)
            st.rerun()

# ---------- Main ----------
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if st.session_state['logged_in']:
        dashboard()
    else:
        login_page()

if __name__ == "__main__":
    main()
