# CloudEagle Country AI Agent

A production-ready agentic workflow built with **LangGraph** and **FastAPI**. This service intelligently parses user queries, fetches live data from the REST Countries API, and synthesizes grounded answers using LLMs.

## Architecture

The agent is designed as a strict state machine with 3 primary nodes:

1. **Intent Extraction Node:** Uses OpenAI Structured Outputs (Pydantic) to extract the target country and data intent. Includes edge-case routing for invalid queries.
2. **API Invocation Node:** Safely queries `restcountries.com` and handles 404/network errors.
3. **Synthesis Node:** Uses the LLM to write a final, conversational response strictly grounded in the fetched API JSON.

## How to Run Locally

1. **Clone the repo and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
