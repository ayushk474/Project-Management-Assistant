import os
import faiss
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEndpoint
from tenacity import retry, wait_fixed, stop_after_attempt
import logging

# Enable Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔹 Load Hugging Face API Key
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not huggingface_api_key:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in environment variables")
# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")  # Use Render's environment variable

if not MONGO_URI:
    raise ValueError("MongoDB URI is missing! Check environment variables.")

client = AsyncIOMotorClient(MONGO_URI)
db = client["Task"]
task_collection = db["Task_data"]

app = FastAPI()

# Request Model
class QueryRequest(BaseModel):
    query: str
    model: str

# Model Mapping
MODEL_MAP = {
    "Hugging Face": "HuggingFaceH4/zephyr-7b-beta"
}

### 🚀 FUNCTION: Fetch Tasks from MongoDB ###
async def fetch_eshway_tasks() -> List[Dict[str, Any]]:
    """Retrieve tasks from MongoDB collection."""
    try:
        tasks = await task_collection.find().to_list(None)
        return tasks if tasks else []
    except Exception as e:
        logger.error(f"MongoDB fetch error: {e}")
        return []

### FUNCTION: Process Text into FAISS Vector Store ###
def process_text(tasks: List[Dict[str, Any]]):
    """Convert task descriptions into embeddings and store in FAISS index."""
    if not tasks:
        return None

    # Format task data
    formatted_texts = [
        f"Title: {task.get('title', 'No Title')}\n"
        f"Description: {task.get('description', 'No Description')}\n"
        f"Status: {task.get('status', 'Unknown')}\n"
        f"Priority: {task.get('priority', 'None')}\n"
        f"Due Date: {task.get('due_date', 'N/A')}"
        f"Assignee Name: {task.get('assignee_name', 'N/A')}"
        for task in tasks
    ]
    full_text = "\n\n".join(formatted_texts)

    # Split text into chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(full_text)

    # Generate embeddings and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    index = faiss.IndexFlatL2(768)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_texts(texts)
    return vector_store

### FUNCTION: Query LLM Model ###
@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def query_model(llm, retriever, query):
    """Query the LLM using FAISS retriever."""
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.invoke({"query": query})


### API Home Route ###
@app.get("/")
def home():
    return {"message": "Welcome to the AI Assistant for Project Management"}

### API Chat Route ###
@app.post("/chat")
async def chat(query_request: QueryRequest):
    """Process user queries with AI-powered responses."""
    
    # Fetch task data from MongoDB
    task_data = await fetch_eshway_tasks()
    if not task_data:
        raise HTTPException(status_code=500, detail="Failed to fetch tasks")

    query_lower = query_request.query.lower()

    # Ensure valid model selection
    if query_request.model not in MODEL_MAP:
        logger.error(f"Invalid model selection: {query_request.model}")
        raise HTTPException(status_code=400, detail="Invalid model selection")

    logger.info(f"Using model: {query_request.model}")

    # Process text for vector search
    vectorstore = process_text(task_data)
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Failed to process task data")
    
    retriever = vectorstore.as_retriever()

    # Initialize LLM with dynamic model selection
   llm = HuggingFaceEndpoint(
        repo_id=MODEL_MAP[query_request.model],  # Dynamically select model
        model_kwargs={"token" : huggingface_api_key},
        temperature=0.6
    )


    # Custom prompt for optimization queries
    if any(keyword in query_lower for keyword in ["how to complete", "faster", "optimize", "decrease project time"]):
        query_prompt = (
            f"Based on the project task data, suggest the best approach to {query_request.query}. "
            "Consider task dependencies, priorities, and deadlines."
        )
        response = query_model(llm, retriever, query_prompt)
    else:
        response = query_model(llm, retriever, query_request.query)

    return {"response": response}

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
