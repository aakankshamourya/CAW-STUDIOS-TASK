import streamlit as st
import pandas as pd
import sqlite3
import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(page_title="Audio Intelligence Platform")

# -------------------------
# LOAD EMBEDDING MODEL
# -------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# -------------------------
# SESSION STATE (VECTOR STORE)
# -------------------------

dimension = 384

if "index" not in st.session_state:
    st.session_state.index = faiss.IndexFlatL2(dimension)
    st.session_state.metadata = {}

index = st.session_state.index
metadata = st.session_state.metadata

# -------------------------
# DATABASE
# -------------------------

conn = sqlite3.connect("songs.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_url TEXT UNIQUE,
    transcript TEXT,
    language TEXT,
    sentiment TEXT,
    intent TEXT,
    themes TEXT,
    energy TEXT
)
""")
conn.commit()

# -------------------------
# MOCK AUDIO PROCESSING
# -------------------------

def process_audio(audio_url):

    if "00004802" in audio_url:
        transcript = "A sad breakup song about heartbreak and tears"
    elif "000099e1" in audio_url:
        transcript = "A high energy party dance track"
    elif "0000ea1f" in audio_url:
        transcript = "A devotional spiritual song about faith"
    else:
        transcript = "A romantic love story song"

    language = "english"
    sentiment = "negative" if "sad" in transcript or "breakup" in transcript else "positive"
    intent = "romantic" if "love" in transcript else "general"
    themes = transcript
    energy = "high" if "party" in transcript else "medium"

    return transcript, language, sentiment, intent, themes, energy


def store_embedding(song_id, text):
    vector = embed_model.encode([text])
    index.add(np.array(vector).astype("float32"))
    metadata[len(metadata)] = song_id


# -------------------------
# UI
# -------------------------

st.title("🎵 Real-Time Audio Intelligence & Conversational Analytics")

# -------------------------
# CSV Upload
# -------------------------

st.header("Upload CSV")

uploaded_file = st.file_uploader("Upload CSV with id,audio_url", type=["csv"])

if uploaded_file:

    # RESET DATABASE & VECTOR STORE
    cursor.execute("DELETE FROM songs")
    conn.commit()

    st.session_state.index = faiss.IndexFlatL2(dimension)
    st.session_state.metadata = {}

    index = st.session_state.index
    metadata = st.session_state.metadata

    df = pd.read_csv(uploaded_file)

    for _, row in df.iterrows():
        audio_url = row["audio_url"]

        transcript, language, sentiment, intent, themes, energy = process_audio(audio_url)

        cursor.execute("""
        INSERT OR IGNORE INTO songs 
        (audio_url, transcript, language, sentiment, intent, themes, energy)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (audio_url, transcript, language, sentiment, intent, themes, energy))

        song_id = cursor.lastrowid
        conn.commit()

        store_embedding(song_id, transcript)

    st.success("Songs Processed Successfully!")
def generate_llm_response(query, songs):
    
    # Build context from retrieved songs
    context = "\n".join([
        f"Song Themes: {song[6]}, Sentiment: {song[4]}, Energy: {song[7]}"
        for song in songs
    ])

    prompt = f"""
    You are an AI music analytics assistant.

    Based on the following retrieved song data:
    {context}

    Answer the user's query:
    {query}

    Provide a concise conversational response.
    """

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]

# -------------------------
# SEARCH
# -------------------------

st.header("Ask About Your Songs")

query = st.text_input("Example: Show me sad songs about heartbreak")

if st.button("Search") and query:

    query_lower = query.lower()

    # -------------------------
    # ANALYTICS QUERIES
    # -------------------------

    if "how many" in query_lower or "count" in query_lower:

        if "sad" in query_lower:
            cursor.execute("SELECT COUNT(*) FROM songs WHERE sentiment='negative'")
        elif "romantic" in query_lower:
            cursor.execute("SELECT COUNT(*) FROM songs WHERE intent='romantic'")
        elif "party" in query_lower:
            cursor.execute("SELECT COUNT(*) FROM songs WHERE energy='high'")
        else:
            cursor.execute("SELECT COUNT(*) FROM songs")

        count = cursor.fetchone()[0]

        prompt = f"""
        The user asked: {query}
        The database result count is: {count}
        Respond conversationally.
        """

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        st.subheader("🧠 AI Analytics Answer")
        st.write(response["message"]["content"])

    # -------------------------
    # LISTING QUERIES
    # -------------------------

    else:

        if "sad" in query_lower:
            cursor.execute("SELECT * FROM songs WHERE sentiment='negative' LIMIT 5")
        elif "romantic" in query_lower:
            cursor.execute("SELECT * FROM songs WHERE intent='romantic' LIMIT 5")
        elif "party" in query_lower:
            cursor.execute("SELECT * FROM songs WHERE energy='high' LIMIT 5")
        else:
            cursor.execute("SELECT * FROM songs LIMIT 5")

        songs = cursor.fetchall()

        if songs:

            # Build context for LLM
            context = "\n".join([
                f"Themes: {song[6]}, Sentiment: {song[4]}, Energy: {song[7]}"
                for song in songs
            ])

            prompt = f"""
            Based on these songs:
            {context}

            Answer the query: {query}
            """

            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )

            st.subheader("🧠 AI Summary")
            st.write(response["message"]["content"])

            st.subheader("🎵 Matching Songs")

            for song in songs:
                st.markdown(f"""
                **🎵 Song ID:** {song[0]}  
                **🔗 URL:** {song[1]}  
                **😊 Sentiment:** {song[4]}  
                **🎯 Intent:** {song[5]}  
                **🏷 Themes:** {song[6]}  
                **⚡ Energy:** {song[7]}  
                ---
                """)

        else:
            st.warning("No matching songs found.")