import os
import faiss
import openai
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_CSV_PATH = "chatbot_2/dataset/index.csv"
CHUNK_CSV_PATH = "chatbot_2/dataset/chunks.csv"
FILES_DIR = "chatbot_2/dataset"
FAISS_DIR = "chatbot_2/faiss_index"
FAISS_PATH = os.path.join(FAISS_DIR, 'faiss.index')

def embed_texts(texts: list[str]) -> list[np.array]:
    # response = openai.Embedding.create(
    #     model=EMBEDDING_MODEL,
    #     input=texts
    # )
    # data = response['data']
    # embeddings = [data[i]['embedding'] for i in data]
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a specific model
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def setup_chunk_index():
    # Read the index file containing document paths and URLs
    index_df = pd.read_csv(INDEX_CSV_PATH)
    print(index_df.columns)
    
    chunks = []
    chunk_index = 0
    
    for _, row in index_df.iterrows():
        file_path = os.path.join(FILES_DIR, str(row['path']))
        url = row['url']
        
        # Read the content of each file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split content into chunks of 500 characters
            for i in range(0, len(content), 500):
                chunk_text = content[i:i+500]
                chunks.append({
                    'chunk_index': chunk_index,
                    'chunk_text': chunk_text,
                    'url': url
                })
                chunk_index += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Create and save chunks dataframe
    chunks_df = pd.DataFrame(chunks)
    os.makedirs(os.path.dirname(CHUNK_CSV_PATH), exist_ok=True)
    chunks_df.to_csv(CHUNK_CSV_PATH, index=False)
    
    return chunks_df

def setup_faiss_index():
    # dimension = 1536  # OpenAI embeddings dimension
    dimension = 384  # SentenceTransformer dimension for 'all-MiniLM-L6-v2'
    index = faiss.IndexFlatL2(dimension)
    
    # Read chunk index
    chunks_df = pd.read_csv(CHUNK_CSV_PATH)
    
    # Embed all chunks
    chunk_texts = chunks_df['chunk_text'].tolist()
    embeddings = embed_texts(chunk_texts)
    
    # Add chunks to faiss
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    
    # Save the FAISS index
    os.makedirs(FAISS_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_DIR, "faiss.index"))

def search_faiss(query: str, top_k=3) -> list[int]:
    index = faiss.read_index(os.path.join(FAISS_DIR, "faiss.index"))
    
    # Embed the query
    query_vec = embed_texts([query])[0]  # Get the first (and only) embedding
    query_vec = np.array([query_vec]).astype('float32')
    
    # Perform search
    distances, indices = index.search(query_vec, top_k)
    
    # Return the matching indices (flattened from 2D array)
    return indices[0].tolist()

def get_matching(indices: list[int]) -> tuple[list[str], list[str]]:
    # Read chunk index
    chunks_df = pd.read_csv(CHUNK_CSV_PATH)
    
    # Get the matching chunks and references
    matching_chunks = []
    matching_refs = []
    
    for idx in indices:
        if 0 <= idx < len(chunks_df):
            chunk_row = chunks_df.iloc[idx]
            matching_chunks.append(chunk_row['chunk_text'])
            matching_refs.append(chunk_row['url'])
    
    return matching_chunks, matching_refs

def search_knowledge_base(query: str):
    indices = search_faiss(query)
    docs, refs = get_matching(indices)
    return docs, refs