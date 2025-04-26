import os
from dotenv import load_dotenv
import openai

from tools.rag_search import CHUNK_CSV_PATH, FAISS_PATH, setup_chunk_index, setup_faiss_index


def init():
    load_dotenv()

    # Set OpenAI API key from environment variable

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise Exception('API KEY NOT SET')


    if not os.path.exists(CHUNK_CSV_PATH):
        setup_chunk_index()
    # Setup FAISS if not exist
    if not os.path.exists(FAISS_PATH):
        setup_faiss_index()