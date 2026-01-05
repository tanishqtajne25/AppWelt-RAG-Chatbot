from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from rag import get_answer  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all websites to connect (Safe for dev)
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],
)

#input
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []  

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Call your existing logic (now inside rag.py)
        response_text = await get_answer(request.message, request.history)
        
        return {"response": response_text}
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)