import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
from chatbot_2.tools.agent import KSUAgent
from chatbot_2.tools.utils import init

init()
# Start FastAPI
app = FastAPI()


# --- Pydantic Models for Request/Response ---
class ChatRequest(BaseModel):
    query: str
    # image_data: Optional[str] = None  # Optional base64 encoded image


class ChatResponse(BaseModel):
    response: str

agent = KSUAgent()

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
    global chats_memory

    try:
        result = agent.get_plan(
            request.query,
            # memory=chats_memory
        )

        # Update global memory with new conversation
        # chats_memory = result["memory"]

        # Return only the response
        return {"response": result["response"]}

    except Exception as e:
        print(f"Error processing request: {e}")  # Log the exception
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/clear")
async def clear_memory():
    """Clear the chatbot's memory"""
    global chats_memory
    chats_memory = []
    return {"message": "chatbot memory cleared successfully"}
