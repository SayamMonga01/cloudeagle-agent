from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.agent import app as agent_app 

app = FastAPI(
    title="CloudEagle AI Agent API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    """
    Executes the LangGraph state machine to answer country-specific queries.
    """
    try:
        result = agent_app.invoke({"user_query": request.query})
        
        if result.get("error"):
            return {"status": "error", "message": result["error"]}
        
        return {
            "status": "success", 
            "country_detected": result.get("country"),
            "intent_detected": result.get("intent"),
            "answer": result.get("final_answer")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal agent execution failed.")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cloudeagle-agent"}