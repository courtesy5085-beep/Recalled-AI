import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from PIL import Image
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import re

# Page configuration
st.set_page_config(
    page_title="Recalled - Your AI Memory Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Card styling */
    .memory-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        border: 1px solid #e0e0e0;
    }
    
    .memory-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px;
    }
    
    /* Success message */
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    
    /* Animated loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        
        .memory-card {
            background: #2d2d44;
            color: white;
            border-color: #3d3d5c;
        }
        
        .success-message {
            background: #1a3d1a;
            color: #90ee90;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('recalled.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  email TEXT,
                  created_at TIMESTAMP)''')
    
    # Memories table
    c.execute('''CREATE TABLE IF NOT EXISTS memories
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Recall events table
    c.execute('''CREATE TABLE IF NOT EXISTS recall_events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  memory_id INTEGER,
                  user_id INTEGER,
                  recalled_at TIMESTAMP,
                  feedback TEXT,
                  FOREIGN KEY (memory_id) REFERENCES memories (id))''')
    
    # Tags table for quick search
    c.execute('''CREATE TABLE IF NOT EXISTS user_tags
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  tag_name TEXT,
                  count INTEGER DEFAULT 1,
                  UNIQUE(user_id, tag_name))''')
    
    conn.commit()
    conn.close()

init_db()

# AI processing functions (simplified for demo)
def generate_ai_summary(content: str, memory_type: str) -> str:
    """Generate AI summary of content"""
    # In production, integrate with OpenAI or other LLM
    words = content.split()
    if len(words) > 50:
        summary = ' '.join(words[:50]) + '...'
    else:
        summary = content
    return f"📝 {summary}"

def detect_emotion(content: str) -> str:
    """Detect emotion from content"""
    emotions = {
        'happy': ['love', 'great', 'amazing', 'wonderful', 'happy'],
        'sad': ['sad', 'sorry', 'unfortunate', 'bad'],
        'exciting': ['exciting', 'awesome', 'fantastic', 'incredible'],
        'inspiring': ['inspire', 'motivate', 'dream', 'future'],
        'useful': ['tip', 'guide', 'tutorial', 'learn', 'how to']
    }
    
    content_lower = content.lower()
    for emotion, keywords in emotions.items():
        if any(keyword in content_lower for keyword in keywords):
            return emotion
    return 'neutral'

def extract_tags(content: str) -> List[str]:
    """Extract relevant tags from content"""
    # Common topics and keywords
    topics = {
        'technology': ['ai', 'code', 'python', 'software', 'app', 'tech'],
        'health': ['fitness', 'workout', 'diet', 'health', 'yoga'],
        'business': ['startup', 'business', 'marketing', 'sales'],
        'travel': ['travel', 'trip', 'hotel', 'flight', 'vacation'],
        'food': ['recipe', 'cooking', 'food', 'restaurant', 'delicious'],
        'education': ['learn', 'course', 'study', 'book', 'research']
    }
    
    detected_tags = set()
    content_lower = content.lower()
    
    for topic, keywords in topics.items():
        if any(keyword in content_lower for keyword in keywords):
            detected_tags.add(topic)
    
    # Add emotion as tag
    emotion = detect_emotion(content)
    if emotion != 'neutral':
        detected_tags.add(emotion)
    
    return list(detected_tags)

# Session management
def login_user(username: str, password: str) -> bool:
    conn = sqlite3.connect('recalled.db')
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", 
              (username, hashed_pw))
    user = c.fetchone()
    conn.close()
    
    if user:
        st.session_state['user_id'] = user[0]
        st.session_state['username'] = username
        st.session_state['logged_in'] = True
        return True
    return False

def register_user(username: str, password: str, email: str) -> bool:
    try:
        conn = sqlite3.connect('recalled.db')
        c = conn.cursor()
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password, email, created_at) VALUES (?, ?, ?, ?)",
                  (username, hashed_pw, email, datetime.now()))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

# Memory management
def save_memory(title: str, content: str, memory_type: str, location: str = ""):
    if 'user_id' not in st.session_state:
        return False
    
    summary = generate_ai_summary(content, memory_type)
    emotion = detect_emotion(content)
    tags = extract_tags(content)
    tags_str = ','.join(tags)
    
    conn = sqlite3.connect('recalled.db')
    c = conn.cursor()
    c.execute('''INSERT INTO memories 
                 (user_id, title, content, memory_type, tags, summary, emotion, location, created_at, last_recalled)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (st.session_state['user_id'], title, content, memory_type, tags_str, summary, emotion, location, 
               datetime.now(), datetime.now()))
    
    memory_id = c.lastrowid
    
    # Update tags count
    for tag in tags:
        c.execute('''INSERT INTO user_tags (user_id, tag_name, count) 
                     VALUES (?, ?, 1) ON CONFLICT(user_id, tag_name) 
                     DO UPDATE SET count = count + 1''', 
                  (st.session_state['user_id'], tag))
    
    conn.commit()
    conn.close()
    return memory_id

def get_smart_recalls(limit: int = 5) -> List[Dict]:
    """Get items to recall based on smart criteria"""
    if 'user_id' not in st.session_state:
        return []
    
    conn = sqlite3.connect('recalled.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Items that haven't been recalled in a while (older than 7 days)
    c.execute('''SELECT *, 
                 julianday('now') - julianday(last_recalled) as days_old,
                 julianday('now') - julianday(created_at) as age_days
                 FROM memories 
                 WHERE user_id = ? AND is_archived = 0
                 ORDER BY days_old DESC, recall_count ASC
                 LIMIT ?''', 
              (st.session_state['user_id'], limit))
    
    recalls = [dict(row) for row in c.fetchall()]
    conn.close()
    return recalls

def log_recall(memory_id: int, feedback: str = ""):
    conn = sqlite3.connect('recalled.db')
    c = conn.cursor()
    
    # Update recall count and last recalled time
    c.execute('''UPDATE memories 
                 SET recall_count = recall_count + 1, 
                     last_recalled = ?
                 WHERE id = ?''',
              (datetime.now(), memory_id))
    
    # Log recall event
    c.execute('''INSERT INTO recall_events (memory_id, user_id, recalled_at, feedback)
                 VALUES (?, ?, ?, ?)''',
              (memory_id, st.session_state['user_id'], datetime.now(), feedback))
    
    conn.commit()
    conn.close()

def search_memories(query: str, tag_filter: str = "", emotion_filter: str = "") -> List[Dict]:
    if 'user_id' not in st.session_state:
        return []
    
    conn = sqlite3.connect('recalled.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    sql = "SELECT * FROM memories WHERE user_id = ? AND is_archived = 0"
    params = [st.session_state['user_id']]
    
    if query:
        sql += " AND (title LIKE ? OR content LIKE ? OR tags LIKE ?)"
        params.extend([f'%{query}%', f'%{query}%', f'%{query}%'])
    
    if tag_filter:
        sql += " AND tags LIKE ?"
        params.append(f'%{tag_filter}%')
    
    if emotion_filter:
        sql += " AND emotion = ?"
        params.append(emotion_filter)
    
    sql += " ORDER BY created_at DESC"
    
    c.execute(sql, params)
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    return results

def get_user_stats():
    if 'user_id' not in st.session_state:
        return {}
    
    conn = sqlite3.connect('recalled.db')
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", 
              (st.session_state['user_id'],))
    total_memories = c.fetchone()[0]
    
    c.execute("SELECT COUNT(DISTINCT tags) FROM memories WHERE user_id = ?", 
              (st.session_state['user_id'],))
    total_tags = c.fetchone()[0]
    
    c.execute("SELECT AVG(recall_count) FROM memories WHERE user_id = ?", 
              (st.session_state['user_id'],))
    avg_recalls = c.fetchone()[0] or 0
    
    c.execute("SELECT emotion, COUNT(*) FROM memories WHERE user_id = ? GROUP BY emotion", 
              (st.session_state['user_id'],))
    emotions = dict(c.fetchall())
    
    conn.close()
    
    return {
        'total_memories': total_memories,
        'total_tags': total_tags,
        'avg_recalls': round(avg_recalls, 1),
        'emotions': emotions
    }

# Main UI components
def login_page():
    st.markdown('<div class="main-header"><h1>🧠 Recalled</h1><p>Your AI-Powered Memory Companion</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if login_user(username, password):
                        st.success(f"Welcome back, {username}! 🎉")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Choose Username")
                new_email = st.text_input("Email")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit_reg = st.form_submit_button("Register", use_container_width=True)
                
                if submit_reg:
                    if new_password != confirm_password:
                        st.error("Passwords don't match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    elif register_user(new_username, new_password, new_email):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists")

def dashboard():
    st.markdown(f'<div class="main-header"><h1>Welcome back, {st.session_state["username"]}! 🧠</h1><p>Your memories are waiting to be recalled</p></div>', unsafe_allow_html=True)
    
    # Stats row
    stats = get_user_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="stat-card"><h3>📚 {stats.get("total_memories", 0)}</h3><p>Total Memories</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="stat-card"><h3>🏷️ {stats.get("total_tags", 0)}</h3><p>Unique Tags</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="stat-card"><h3>🔄 {stats.get("avg_recalls", 0)}</h3><p>Avg Recalls</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="stat-card"><h3>📊 {len(stats.get("emotions", {}))}</h3><p>Emotion Types</p></div>', unsafe_allow_html=True)
    
    # Smart recall section
    st.markdown("## 🔮 Today's Smart Recalls")
    st.markdown("*Memories you might want to revisit*")
    
    recalls = get_smart_recalls(3)
    
    if recalls:
        for recall in recalls:
            with st.container():
                st.markdown(f"""
                <div class="memory-card">
                    <h3>📌 {recall['title']}</h3>
                    <p><strong>Type:</strong> {recall['memory_type']} | 
                    <strong>Emotion:</strong> {recall['emotion']} | 
                    <strong>Age:</strong> {int((datetime.now() - datetime.fromisoformat(recall['created_at'])).days)} days old</p>
                    <p>{recall['summary']}</p>
                    <p><strong>Tags:</strong> #{' #'.join(recall['tags'].split(','))}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button(f"👍 Recall", key=f"recall_{recall['id']}"):
                        log_recall(recall['id'], "manual")
                        st.success("Memory recalled! 🎉")
                with col2:
                    if st.button(f"🔍 View Full", key=f"view_{recall['id']}"):
                        st.session_state['view_memory'] = recall['id']
    else:
        st.info("No memories to recall yet. Start saving your thoughts! 📝")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ New Memory", use_container_width=True):
            st.session_state['show_new_memory'] = True
    
    with col2:
        if st.button("🔍 Explore Memories", use_container_width=True):
            st.session_state['show_explore'] = True
    
    # New memory form
    if st.session_state.get('show_new_memory', False):
        st.markdown("### ✨ Create New Memory")
        
        with st.form("new_memory_form"):
            title = st.text_input("Title")
            memory_type = st.selectbox("Memory Type", ["Article", "Note", "Idea", "Quote", "Link", "Other"])
            content = st.text_area("Content", height=150)
            location = st.text_input("Location (optional)", placeholder="e.g., Home, Office, Cafe")
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("💾 Save Memory", use_container_width=True)
            with col2:
                cancel = st.form_submit_button("Cancel", use_container_width=True)
            
            if submitted and title and content:
                memory_id = save_memory(title, content, memory_type, location)
                if memory_id:
                    st.success("Memory saved successfully! ✨")
                    st.session_state['show_new_memory'] = False
                    st.rerun()
            elif submitted:
                st.warning("Please fill in both title and content")
        
        if cancel:
            st.session_state['show_new_memory'] = False
            st.rerun()
    
    # Explore section
    if st.session_state.get('show_explore', False):
        st.markdown("### 🔍 Explore Your Memories")
        
        # Search and filters
        col1, col2, col3 = st.columns(3)
        with col1:
            search_query = st.text_input("🔎 Search", placeholder="Type to search...")
        with col2:
            all_tags = ["All"] + list(stats.get('emotions', {}).keys())
            emotion_filter = st.selectbox("🎭 Emotion Filter", all_tags)
        with col3:
            tag_list = ["All", "technology", "health", "business", "travel", "food", "education", "happy", "sad", "exciting", "inspiring", "useful"]
            tag_filter = st.selectbox("🏷️ Tag Filter", tag_list)
        
        # Apply filters
        if emotion_filter == "All":
            emotion_filter = ""
        if tag_filter == "All":
            tag_filter = ""
        
        memories = search_memories(search_query, tag_filter, emotion_filter)
        
        if memories:
            st.markdown(f"**Found {len(memories)} memories**")
            
            for memory in memories:
                with st.expander(f"📌 {memory['title']} - {memory['created_at'][:10]}"):
                    st.markdown(f"**Type:** {memory['memory_type']}")
                    st.markdown(f"**Emotion:** {memory['emotion']}")
                    st.markdown(f"**Content:** {memory['content'][:200]}...")
                    st.markdown(f"**Tags:** #{' #'.join(memory['tags'].split(','))}")
                    st.markdown(f"**Recalls:** {memory['recall_count']} times")
                    
                    if st.button(f"Recall this memory", key=f"recall_explore_{memory['id']}"):
                        log_recall(memory['id'], "explore")
                        st.success("Memory recalled! ✨")
                        st.rerun()
        else:
            st.info("No memories f
