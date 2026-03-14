import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384
index = faiss.IndexFlatL2(dimension)

metadata = {}

def embed_and_store(song_id, text):
    vector = model.encode([text])
    index.add(np.array(vector).astype("float32"))
    metadata[len(metadata)] = song_id

def search(query, k=5):
    q_vector = model.encode([query])
    D, I = index.search(np.array(q_vector).astype("float32"), k)
    return [metadata[i] for i in I[0]]
if __name__ == "__main__":
    
    # Add sample data
    embed_and_store(1, "This is a romantic love song full of passion")
    embed_and_store(2, "This is a sad breakup heartbreak song")
    embed_and_store(3, "High energy party dance song")

    # Search
    results = search("sad songs")
    print("Search Results:", results)