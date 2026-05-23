# 🧠 Recalled AI

Production-grade AI memory assistant built with Streamlit.

## Features

- Semantic memory search
- AI summaries/tags/emotions
- Audio transcription
- Image captioning
- RAG chat over memories
- Weekly insights
- Export/import
- Timeline UI

## Setup

### 1. Clone

git clone <repo>

### 2. Install

pip install -r requirements.txt

### 3. Add .env

OPENAI_API_KEY=your_key

### 4. Run

streamlit run app.py

## Streamlit Cloud Notes

- Uses ChromaDB instead of FAISS
- No C++ FAISS dependency
- SQLite + DuckDB compatible
- Graceful fallback if ffmpeg missing
