import os
import faiss
import openai
import numpy as np
import pandas as pd

EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_CSV_PATH = "dataset/index.csv"
FILES_DIR = "dataset/files"
FAISS_DIR = "faiss_index"

def embed_text(text):
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response['data'][0]['embedding']

def setup_faiss_index():
    dimension = 1536
    index = faiss.IndexFlatL2(dimension)
    texts = []

    for filename in os.listdir(FILES_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(FILES_DIR, filename), "r", encoding="utf-8") as f:
                content = f.read()
                chunks = [content[i:i+500] for i in range(0, len(content), 500)]
                for chunk in chunks:
                    vector = embed_text(chunk)
                    index.add(np.array([vector], dtype=np.float32))
                    texts.append(chunk)

    os.makedirs(FAISS_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_DIR, "faiss.index"))
    with open(os.path.join(FAISS_DIR, "texts.txt"), "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.replace("\n", " ") + "\n")

def search_faiss(query, top_k=3):
    index = faiss.read_index(os.path.join(FAISS_DIR, "faiss.index"))
    with open(os.path.join(FAISS_DIR, "texts.txt"), "r", encoding="utf-8") as f:
        texts = f.read().splitlines()

    query_vec = embed_text(query)
    D, I = index.search(np.array([query_vec], dtype=np.float32), top_k)

    return [texts[i] for i in I[0]]

def search_references(query):
    refs = pd.read_csv(INDEX_CSV_PATH).to_dict(orient="records")
    matches = []
    query = query.lower()

    for ref in refs:
        content = str(ref.get("content", "")).lower()
        if any(word in content for word in query.split()):
            matches.append(f"📚 {ref.get('source', '')}: {ref.get('content', '')}")

    return matches[:3]

def search_knowledge_base(query):
    docs = search_faiss(query)
    refs = search_references(query)
    return docs, refs
