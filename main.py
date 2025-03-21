import os
import faiss
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
import asyncio

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


MODEL_MAP = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "Hugging Face": "HuggingFaceH4/zephyr-7b-alpha"
}

# Request Model
class QueryRequest(BaseModel):
    query: str
    model: str  # Specify which LLM model to use

# FUNCTION: Fetch Tasks from MongoDB
async def fetch_eshway_tasks() -> List[Dict[str, Any]]:
    """Retrieve tasks from MongoDB collection."""
    try:
        tasks = await task_collection.find({}, {"title": 1, "description": 1, "status": 1, "priority": 1, "due_date": 1, "assignee_name": 1}).to_list(None)
        return tasks if tasks else []
    except Exception as e:
        logger.error(f"MongoDB fetch error: {e}")
        return []

# FUNCTION: Process Text into FAISS Vector Store
def process_text(tasks: List[Dict[str, Any]]):
    """Convert task descriptions into embeddings and store in FAISS index."""
    if not tasks:
        logger.warning("No tasks found to process.")
        return None

    # 🔹 Format task data
    formatted_texts = [
        f"Title: {task.get('title', 'No Title')}\n"
        f"Description: {task.get('description', 'No Description')}\n"
        f"Status: {task.get('status', 'Unknown')}\n"
        f"Priority: {task.get('priority', 'None')}\n"
        f"Due Date: {task.get('due_date', 'N/A')}\n"
        f"Assignee Name: {task.get('assignee_name', 'N/A')}"
        for task in tasks
    ]
    full_text = "\n\n".join(formatted_texts)

    # 🔹 Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(full_text)

    # 🔹 Generate embeddings and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    d = 384  # Embedding dimension
    index = faiss.IndexFlatL2(d)  # Flat L2 index (optimized for similarity search)

    # Ensure index training if needed
    if not index.is_trained:
        index.train(np.random.rand(100, d).astype(np.float32))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_texts(texts)

    logger.info(f"FAISS VectorStore contains {len(texts)} embedded texts.")
    return vector_store

async def stream_query_model(llm, retriever, query):
    """Stream LLM response using FAISS retriever and provide task context."""
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # 🔹 Retrieve relevant task descriptions
    similar_docs = retriever.invoke(query)
    relevant_info = "\n".join([doc.page_content for doc in similar_docs[:3]])

    if not relevant_info:
        relevant_info = "No relevant task data found."

    # 🔹 Optimized query prompt
    prompt = (
        f"Here are some relevant project tasks:\n{relevant_info}\n\n"
        f"Based on this information, answer the query: {query}"
    )

    response = qa.invoke({"query": prompt})

    logger.info(f"Raw LLM Response: {response}")

    # Streaming Response (Yield word-by-word)
    for word in response.get("result", "No response generated").split():
        yield word + " "
        await asyncio.sleep(0.15)  # Simulate real-time streaming
    )

# API Home Route
@app.get("/")
def home():
    return {"message": "Welcome to AI-Driven Project Management Assistant"}

# API Chat Route with Streaming
@app.post("/chat")
async def chat(query_request: QueryRequest):
    """Process user queries, including pending tasks, completion strategies, etc., with streaming."""

    # 🔹 Fetch task data
    task_data = await fetch_eshway_tasks()
    if not task_data:
        raise HTTPException(status_code=500, detail="Failed to fetch tasks")

    # 🔹 Count Pending Tasks
    pending_tasks = [task for task in task_data if task.get("status", "").lower() == "todo"]
    num_pending_tasks = len(pending_tasks)

    # 🔹 Ensure valid model selection
    if query_request.model not in MODEL_MAP:
        logger.error(f"Invalid model selection: {query_request.model}")
        raise HTTPException(status_code=400, detail="Invalid model selection")

    logger.info(f"Using model: {query_request.model}")

    # 🔹 Process text for vector search
    vectorstore = process_text(task_data)
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Failed to process task data")

    retriever = vectorstore.as_retriever()

    # 🔹 Initialize LLM
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_MAP[query_request.model],
        model_kwargs={"token": huggingface_api_key},
        temperature=0.6
    )

    # 🔹 Modify Query with Pending Task Data
    query_with_context = (
        f"There are {num_pending_tasks} pending tasks.\n"
        f"Here are their details:\n"
        + "\n".join([f"- {task['title']} (Due: {task.get('due_date', 'N/A')})" for task in pending_tasks])
        + f"\nNow answer the query: {query_request.query}"
    )

    # 🔹 Return Streaming Response
    return StreamingResponse(stream_query_model(llm, retriever, query_with_context), media_type="text/plain")


# Run FastAPI

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
