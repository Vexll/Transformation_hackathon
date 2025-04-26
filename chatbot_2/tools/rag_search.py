import os
import pickle
import time
import faiss
import openai
import numpy as np
import pandas as pd
import zipfile
from sentence_transformers import SentenceTransformer
import tiktoken
from tqdm import tqdm  # OpenAI's tokenizer library

EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_CSV_PATH = "chatbot_2/dataset/index.csv"
CHUNK_CSV_PATH = "chatbot_2/dataset/chunks.csv"
FILES_DIR = "chatbot_2/dataset"
FAISS_DIR = "chatbot_2/dataset"
FAISS_PATH = os.path.join(FAISS_DIR, 'faiss.index')
ZIP_FILE_PATH = os.path.join(FILES_DIR, 'files.zip')  # Path to your zip file
CHUNK_SIZE_CHARS = 5000

def embed_query(text: str) -> list[np.array]:
    response = openai.Embedding.create(
    model=EMBEDDING_MODEL,
    input=text
    )
    data = response['data']
    embeddings = [item['embedding'] for item in data]
    return embeddings

def embed_texts(texts: list[str]) -> list[np.array]:
    print('START EMBED')
    embeddings = []
    start_index = 0
    if os.path.exists(FILES_DIR+"/embeddings.pkl"):
        with open(FILES_DIR+"/embeddings.pkl", 'rb') as f:
            embeddings = pickle.load(f)
            start_index = len(embeddings)
            print(f"EMBEEDDING START INDEX={start_index}")
            # comment this later
            return embeddings
    
    max_batch_size = 1024
    for i in range(start_index, len(texts), max_batch_size):
        print(f"{i}/{len(texts)}")
        temp_texts = texts[i:i+max_batch_size]
        response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=temp_texts
        )
        data = response['data']
        temp_embeddings = [item['embedding'] for item in data]
        embeddings.extend(temp_embeddings)
        
        # write embeddings to file
        with open(FILES_DIR+"/embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
        # wait
        time.sleep(60)
    # model = SentenceTransformer('all-MiniLM-L6-v2') # Using a specific model
    # embeddings = model.encode(texts, show_progress_bar=True)
    print('END EMBED')
    return embeddings

def chunk_text(text, max_tokens=800, overlap=50) -> list[str]:
    """
    Splits text into chunks for embedding.
    
    Args:
        text (str): The full input text.
        max_tokens (int): Maximum tokens per chunk.
        overlap (int): Overlapping tokens between chunks.
    
    Returns:
        List[str]: List of text chunks.
    """
    encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(encoding.decode(chunk))
        start += max_tokens - overlap  # move window forward with overlap
    return chunks


def setup_chunk_index():
    print('START CHUNKING')
    index_df = pd.read_csv(INDEX_CSV_PATH)
    print(index_df.columns)

    chunks = []
    chunk_index = 0

    try:
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zf:
            for _, row in tqdm(index_df.iterrows()):
                file_path_in_zip = str(row['path'])
                url = row['url']
                try:
                    with zf.open(file_path_in_zip.replace("\\", '/'), 'r') as f:
                        content = f.read().decode('utf-8')
        
                    for c in chunk_text(content):
                        chunks.append({
                            'chunk_index': chunk_index,
                            'chunk_text': c,
                            'url': url
                        })
                        chunk_index += 1
                except KeyError:
                    print(f"Error: File '{file_path_in_zip}' not found in the zip archive.")
                except Exception as e:
                    print(f"Error processing file '{file_path_in_zip}' from zip: {e}")
    except FileNotFoundError:
        print(f"Error: Zip file not found at '{ZIP_FILE_PATH}'.")
        return None
    except Exception as e:
        print(f"Error opening or processing the zip file: {e}")
        return None
    
    print(f'CHUNKS: {chunk_index}')
    chunks_df = pd.DataFrame(chunks)
    os.makedirs(os.path.dirname(CHUNK_CSV_PATH), exist_ok=True)
    chunks_df.to_csv(CHUNK_CSV_PATH, index=False)

    return chunks_df

def setup_faiss_index():
    dimension = 1536  # OpenAI embeddings dimension
    # dimension = 384
    index = faiss.IndexFlatL2(dimension)

    chunks_df = pd.read_csv(CHUNK_CSV_PATH)

    chunk_texts = chunks_df['chunk_text'].tolist()
    chunk_texts = [str(t) for t in chunk_texts]
    embeddings = embed_texts(chunk_texts)

    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)

    os.makedirs(FAISS_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_PATH)

def search_faiss(query: str, top_k=5) -> list[int]:
    index = faiss.read_index(FAISS_PATH)

    query_vec = embed_query(query)[0]
    query_vec = np.array([query_vec]).astype('float32')

    distances, indices = index.search(query_vec, top_k)

    return indices[0].tolist()

def get_matching(indices: list[int]) -> tuple[list[str], list[str]]:
    chunks_df = pd.read_csv(CHUNK_CSV_PATH)

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
