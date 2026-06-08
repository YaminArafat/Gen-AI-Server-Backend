from __future__ import annotations

import os
from pathlib import Path
import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from server.gen_ai_service import execute_async_pipeline_job, get_orchestration_chain
from server.task_manager import global_task_manager, TaskState
from server.rag_vector_store import build_vector_store, chroma_collection_exists, query_rag
from server.r_and_d_code.speech_to_text_fw import speech_to_text
from server.celery_app import process_config_generation, celery_engine

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

async def lifespan(app: FastAPI):
    print("Application startup: Initializing resources...")
    initialize_rag_vector_store()
    yield
    print("Application shutdown: Cleaning up resources...")

app = FastAPI(
    title="Text-to-JSON Config Generation AI Service",
    description="Production-grade Multimodal LangChain Execution Engine with local RAG.",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)
tasks: dict[str, dict] = {}
lock = threading.Lock()
gpu_lock = threading.Lock()

class ConfigRequest(BaseModel):
    inputText: str
    useRAG: bool = True

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

def update_task_status(task_id: str, status: str, progress: int = 0, result=None, error: str | None = None):
    with lock:
        tasks[task_id] = {
            "status": status,
            "progress": progress,
            "result": result,
            "error": error,
        }

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "config-generation-ai"}
    
@app.post("/api/v4/config/text", status_code=202)
async def config_from_text(request: ConfigRequest) -> dict:
    if not request.inputText.strip():
        raise HTTPException(status_code=400, detail="inputText is required")
    
    context = "No additional context provided."
    docs = []
    if request.useRAG:
        try:
            docs = query_rag(request.inputText, k=4)
            context = "\n\n".join(doc["page_content"] for doc in docs)
        except Exception as e:
            pass

    
    job_payload = {
        "user_input": request.inputText.strip(),
        "context": context,
        "use_rag": request.useRAG
    }
    async_job = process_config_generation.delay(job_payload)    

    return {
        "task_id": async_job.id,
        "status": "Queued",
        "message": "Task queued for asynchronous compilation."
    }

@app.post("/api/v4/config/audio", status_code=202)
async def config_from_audio(audio: UploadFile = File(...), useRAG: bool = True) -> dict:
    if not audio:
        raise HTTPException(status_code=400, detail="Audio file payload is required.")

    text = speech_to_text(audio)
    if not text:
        raise HTTPException(status_code=500, detail="Audio transcription failure.")

    context = "No additional context provided."
    docs = []
    if useRAG:
        try:
            docs = query_rag(text, k=4)
            context = "\n\n".join(doc["page_content"] for doc in docs)
        except Exception as e:
            pass

    
    job_payload = {
        "user_input": text.strip(),
        "context": context,
        "use_rag": useRAG
    }
    async_job = process_config_generation.delay(job_payload)    

    return {
        "task_id": async_job.id,
        "status": "Queued",
        "message": "Task queued for asynchronous compilation."
    }

def initialize_rag_vector_store():
    print("Checking RAG Vector Store status...")
    try:
        if chroma_collection_exists():
            print("Existing RAG Vector Store found. Initialization skipped.")
            return
        else:
            store = build_vector_store()
            store_info = store._collection.get() if hasattr(store, "_collection") else {}
            document_count = len(store_info.get("ids", []))
            
            if document_count == 0:
                print("Vector store is empty.")
            else:
                print(f"Persistent Vector Store initialized with {document_count} documents")
            
    except Exception as e:
        print(f"Failed to initialize RAG Vector store on startup: {e}")

@app.post("/api/v1/knowledge/ingest")
def knowledge_ingest() -> dict:
    store = build_vector_store()
    store_info = store._collection.get() if hasattr(store, "_collection") else {}
    return {
        "message": "Vector store created",
        "persist_directory": str(Path(os.getenv("CHROMA_PERSIST_DIR", ROOT_DIR / "server" / "chroma_db"))),
        "documents": len(store_info.get("ids", [])),
    }

@app.post("/api/v1/knowledge/query")
def knowledge_query(request: QueryRequest) -> dict:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    docs = query_rag(request.query, k=request.top_k)
    return {"query": request.query, "results": docs}

def _run_config_task(task_id: str, user_input: str, use_rag: bool):
    # update_task_status(task_id, "Queued", 0)
    try:
        update_task_status(task_id, "Running", 10)

        context = "No additional context provided."
        docs = []

        if use_rag:
            update_task_status(task_id, "Retrieving RAG context", 15)
            docs = query_rag(user_input, k=4)
            context = "\n\n".join(doc["page_content"] for doc in docs)
            update_task_status(task_id, "RAG ready", 30)

        with gpu_lock:
            update_task_status(task_id, "Invoking model", 40)

            def _progress_cb(status: str, progress: int, result: dict | None = None, error: str | None = None):
                try:
                    update_task_status(task_id, status, progress, result, error)
                except Exception:
                    pass

            chain = get_orchestration_chain(task_id=task_id, progress_callback=_progress_cb)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            pipeline_output = loop.run_until_complete(
                chain.ainvoke({"user_input": user_input, "context": context})
            )
            loop.close()

        update_task_status(task_id, "Finalizing output", 85)

        update_task_status(
            task_id,
            "Completed",
            100,
            result={
                "config": pipeline_output["config"],
                "generated_asset_path": pipeline_output["generated_asset_path"],
                "rag": {"enabled": use_rag, "documents": docs},
            },
        )
    except Exception as exc:
        update_task_status(task_id, "Failed", 100, error=str(exc))

@app.get("/api/v2/tasks/{task_id}")
async def get_task_status(task_id: str):
    job_query = celery_engine.AsyncResult(task_id)
    if job_query.state == "PENDING":
        return {"task_id": task_id, "status": "Pending", "result": None}
    elif job_query.state == "SUCCESS":
        return {"task_id": task_id, "status": "Completed", "result": job_query.result}
    elif job_query.state == "FAILURE":
        return {"task_id": task_id, "status": "Failed", "error": str(job_query.info)}
        
    return {"task_id": task_id, "status": job_query.state, "result": None}

async def continuous_queue_worker():
    print("Background Execution Queue Worker Booted Successfully.")
    while True:
        try:
            job_pack = await global_task_manager.dequeue_job()
            task_id = job_pack["task_id"]
            payload = job_pack["payload"]
            
            asyncio.create_task(execute_async_pipeline_job(task_id, payload))
            
            global_task_manager.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Queue Worker Cluster encountered an error: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(continuous_queue_worker())

@app.get("/api/v1/tasks/{task_id}", response_model=TaskState)
async def get_task_status(task_id: str):
    state = global_task_manager.get_task_status(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Requeste task id not found.")
    return state

@app.get("/api/v1/task/status/{task_id}")
def task_status(task_id: str):
    with lock:
        task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Invalid task ID")
    return task

@app.get("/api/v1/task/result/{task_id}")
def task_result(task_id: str):
    with lock:
        task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Invalid task ID")
    if task["status"] != "Completed":
        return {"status": task["status"], "progress": task["progress"]}
    return task

@app.post("/api/v2/config/audio")
def config_from_audio(audio: UploadFile = File(...), useRAG: bool = True) -> dict:
    if not audio:
        raise HTTPException(status_code=400, detail="audio file is required")
    task_id = str(uuid.uuid4())
    update_task_status(task_id, "Queued", 0)

    text = speech_to_text(audio)
    if not text:
        raise HTTPException(status_code=500, detail="Failed to transcribe audio file payload")

    update_task_status(task_id, "Running", 10)
    executor.submit(_run_config_task, task_id, text.strip(), useRAG)

    return {"task_id": task_id, "status": "Queued"}

@app.post("/api/v2/config/text")
def config_from_text(request: ConfigRequest):
    if not request.inputText.strip():
        raise HTTPException(status_code=400, detail="inputText is required")

    task_id = str(uuid.uuid4())
    update_task_status(task_id, "Queued", 0)
    executor.submit(_run_config_task, task_id, request.inputText.strip(), request.useRAG)

    return {"task_id": task_id, "status": "Queued"}

@app.post("/api/v3/config/text", status_code=202)
async def config_from_text(request: ConfigRequest) -> dict:
    if not request.inputText.strip():
        raise HTTPException(status_code=400, detail="inputText is required")

    task_id = global_task_manager.create_task()
    
    job_payload = {
        "user_input": request.inputText.strip(),
        "use_rag": request.useRAG
    }
    await global_task_manager.enqueue_job(task_id, job_payload)
    
    return {
        "task_id": task_id,
        "status": "Queued",
        "message": "Task queued for asynchronous compilation."
    }

@app.post("/api/v1/config/text")
async def config_from_text(request: ConfigRequest) -> dict:
    pipeline_chain = get_orchestration_chain()

    if not request.inputText.strip():
        raise HTTPException(status_code=400, detail="inputText is required")

    context = "No additional context provided."
    docs = []
    if request.useRAG:
        try:
            docs = query_rag(request.inputText, k=4)
            context = "\n\n".join(doc["page_content"] for doc in docs)
        except Exception:
            context = "No additional context provided."

    try:
        pipeline_output = await pipeline_chain.ainvoke({
            "user_input": request.inputText.strip(),
            "context": context
        })
        
        return {
            "config": pipeline_output["config"],
            "generated_asset_path": pipeline_output["generated_asset_path"],
            "rag": {"enabled": request.useRAG, "documents": docs}
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LangChain Execution Failure: {exc}")

@app.post("/api/v1/config/audio")
async def config_from_audio(audio: UploadFile = File(...), useRAG: bool = True) -> dict:
    pipeline_chain = get_orchestration_chain()

    if not audio:
        raise HTTPException(status_code=400, detail="audio file is required")

    text = speech_to_text(audio)
    if not text:
        raise HTTPException(status_code=500, detail="Failed to transcribe audio file payload")

    context = "No additional context provided."
    docs = []
    if useRAG:
        try:
            docs = query_rag(text, k=4)
            context = "\n\n".join(doc["page_content"] for doc in docs)
        except Exception:
            context = "No additional context provided."

    try:
        pipeline_output = await pipeline_chain.ainvoke({
            "user_input": text.strip(),
            "context": context
        })
        
        return {
            "transcript": text,
            "config": pipeline_output["config"],
            "generated_asset_path": pipeline_output["generated_asset_path"],
            "rag": {"enabled": useRAG, "documents": docs}
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LangChain Audio Pipeline Execution Failure: {exc}")

@app.post("/api/v3/config/audio", status_code=202)
async def config_from_audio(audio: UploadFile = File(...), useRAG: bool = True) -> dict:
    if not audio:
        raise HTTPException(status_code=400, detail="Audio file payload is required.")

    text = speech_to_text(audio)
    if not text:
        raise HTTPException(status_code=500, detail="Audio transcription failure.")

    task_id = global_task_manager.create_task()
    
    job_payload = {
        "user_input": text.strip(),
        "use_rag": useRAG
    }
    await global_task_manager.enqueue_job(task_id, job_payload)
    
    return {
        "task_id": task_id,
        "transcript": text,
        "status": "Queued",
        "message": "Audio payload transcribed and execution task queued successfully."
    }
