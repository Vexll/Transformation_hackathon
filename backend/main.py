import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
from ksu_chatbot.main import PreorderAgent

# from dotenv import load_dotenv

# load_dotenv()
# openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = "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"

# --- Instantiate the Chatbot Agent ---
# This is created once when the server starts
preorder_chatbot = PreorderAgent()

# Start FastAPI
app = FastAPI()


# --- Pydantic Models for Request/Response ---
class ChatRequest(BaseModel):
    query: str
    image_data: Optional[str] = None  # Optional base64 encoded image


class ChatResponse(BaseModel):
    response: str


# Store conversation memory for each chatbot
preorder_memory = []


# --- API Endpoint ---
@app.post("/ksu-chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Process a user query using the preorder chatbot.

    - **query**: The user's message.
    - **image_data**: Optional base64 encoded image.
    """
    global preorder_memory

    try:
        result = preorder_chatbot.process_order(
            query=request.query, memory_input=preorder_memory
        )

        # Update global memory with new conversation
        preorder_memory = result["memory"]

        # Return only the response
        return {"response": result["response"]}

    except Exception as e:
        print(f"Error processing request: {e}")  # Log the exception
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/clear")
async def clear_memory():
    """Clear the chatbot's memory"""
    global preorder_memory
    global report_memory
    preorder_memory = []
    report_memory = []
    return {"message": "chatbot memory cleared successfully"}
