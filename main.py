from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot_2.tools.agent import KSUAgent
from chatbot_2.tools.utils import init

init()

agent = KSUAgent()
print('starting...')
# Start FastAPI
app = FastAPI()


# --- Pydantic Models for Request/Response ---
class ChatRequest(BaseModel):
    query: str
    # image_data: Optional[str] = None  # Optional base64 encoded image


class ChatResponse(BaseModel):
    response: str


# --- API Endpoint ---
@app.post("/ksu-chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Process a user query using the preorder chatbot.

    - **query**: The user's message.
    - **image_data**: Optional base64 encoded image.
    """
    try:
        plan = agent.get_plan(request.query)
        response = agent.execute_plan(request.query, plan)
        print(f"RESPONSE: {response}")
        

        # Return only the response
        return {"response": response}

    except Exception as e:
        print(f"Error processing request: {e}")  # Log the exception
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/clear")
async def clear_memory():
    """Clear the chatbot's memory"""
    agent.memory.memory = []
    return {"message": "chatbot memory cleared successfully"}
