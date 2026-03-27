import os
import requests
import logging
from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Configure production logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    user_query: str
    country: Optional[str]
    intent: Optional[List[str]]
    api_data: Optional[dict]
    final_answer: Optional[str]
    error: Optional[str]

class IntentExtraction(BaseModel):
    is_valid_country_query: bool = Field(description="True if asking about a real country.")
    country: str = Field(description="The country name extracted.", default="")
    intent: List[str] = Field(description="Data requested (e.g., 'population', 'capital').", default_factory=list)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def extract_intent_node(state: AgentState):
    query = state.get("user_query")
    logger.info(f"Extracting intent for query: {query}")
    
    try:
        structured_llm = llm.with_structured_output(IntentExtraction)
        prompt = f"Analyze this user query and extract the country and their intent: {query}"
        result = structured_llm.invoke(prompt)
        
        if not result.is_valid_country_query:
            logger.warning("Invalid country query detected.")
            return {"error": "Invalid query. Please ask a specific question about a real country."}
        
        return {"country": result.country.lower(), "intent": result.intent}
    
    except Exception as e:
        logger.error(f"Intent extraction failed: {str(e)}")
        return {"error": "Failed to parse query intent."}

def fetch_country_data_node(state: AgentState):
    country = state.get("country")
    logger.info(f"Fetching REST data for: {country}")
    
    try:
        response = requests.get(f"https://restcountries.com/v3.1/name/{country}")
        if response.status_code == 404:
            return {"error": f"Data not found for country: {country}."}
        
        response.raise_for_status()
        return {"api_data": response.json()[0]}
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return {"error": "External API connection failed."}

def synthesize_answer_node(state: AgentState):
    logger.info("Synthesizing final answer")
    
    prompt = f"""
    Answer the user's query using ONLY the provided JSON API data. Keep it concise.
    Query: {state.get("user_query")}
    Data: {state.get("api_data")}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_answer": response.content}

def route_after_intent(state: AgentState):
    return END if state.get("error") else "fetch_data"

def route_after_fetch(state: AgentState):
    return END if state.get("error") else "synthesize_answer"

# Graph Compilation
workflow = StateGraph(AgentState)
workflow.add_node("extract_intent", extract_intent_node)
workflow.add_node("fetch_data", fetch_country_data_node)
workflow.add_node("synthesize_answer", synthesize_answer_node)

workflow.add_edge(START, "extract_intent")
workflow.add_conditional_edges("extract_intent", route_after_intent)
workflow.add_conditional_edges("fetch_data", route_after_fetch)
workflow.add_edge("synthesize_answer", END)

app = workflow.compile()