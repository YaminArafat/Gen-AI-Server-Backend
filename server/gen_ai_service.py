from __future__ import annotations

import json
import os
import io
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Callable

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.string import StringPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableConfig
from ollama import chat as ollama_chat
from server.background_image_generate import background_image_generator
from server.schemas import Config
from server.task_manager import global_task_manager
from server.rag_vector_store import query_rag

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor

from server.telemetry_db import log_telemetry
from services.s3_service import storage_service

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# s3_bucket_name = os.getenv("MINIO_ASSETS_BUCKET_NAME", "ai-assets")

SYSTEM_PROMPT = """You are an expert wearable device UI/UX configuration engine. 
Your sole task is to generate valid architectural JSON layout models for smartwatches based on user text queries.
You must absolutely conform your reasoning to match the format of the examples provided below.
Strict Rules for Image and Media Generation:
1. If the user asks for a visual theme, background, or specific object image (e.g., "A cat image", "a snowy mountain background"), you MUST delegate this to the media generation pipeline:
   - Set `MediaRequired` to 'STATIC_PNG'.
   - Set `MediaPrompt` to a descriptive visual prompt for the asset (e.g., "cat", "snowy mountain").

2. ONLY set `MediaRequired` to 'ANIMATED_GIF' if the user explicitly requests motion-based keywords like "animated", "moving", "gif", "video", or "looping".

3. Set `MediaRequired` to 'NONE' ONLY if the user does not ask for any custom illustrations, objects, images, or scenery backgrounds (e.g., "make the background solid black with red text").
"""

def initialize_llm_observability():
    otlp_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/v1/traces")
    
    provider = TracerProvider()
    processor = SimpleSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    LangChainInstrumentor().instrument()
    print(f"Enterprise LLM Observability Tracing hooked to OTLP: {otlp_endpoint}")

# if os.getenv("ENABLE_OBSERVABILITY", "true").lower() == "true":
#     initialize_llm_observability()

def _load_prompt() -> str:
    prompt_path = ROOT_DIR / "datasets" / "rag_datasets" / "updated_system_prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return "You are an expert JSON config device engine. Parse user requests into required JSON schemas."


def _build_prompt_template(parser: PydanticOutputParser) -> ChatPromptTemplate:
    system_prompt = SystemMessagePromptTemplate(
        prompt=StringPromptTemplate(
            input_variables=["base_prompt", "format_instructions"],
            template="{base_prompt}\n\n{format_instructions}",
        )
    )
    human_prompt = HumanMessagePromptTemplate(
        prompt=StringPromptTemplate(
            input_variables=["user_input", "context"],
            template="User request: {user_input}\n\nContext:\n{context}",
        )
    )
    return ChatPromptTemplate.from_messages([system_prompt, human_prompt])


def _create_ollama_llm() -> Ollama:
    return Ollama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_k_m"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.5")),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
    )

async def _run_stable_diffusion(prompt: str, task_id: str | None) -> tuple[io.BytesIO, str]:
    print(f"[GPU Workload] Stable Diffusion serving prompt: '{prompt}'")
    local_out_file = f"/server/static/docker_generated_outputs/{task_id}_ai_generated_{prompt}.png"
    global_task_manager.update_task(
                task_id,
                status="Running Stable Diffusion",
                progress=50
            )
    file_stream, generated_asset_path = background_image_generator.generate_background_image_webUI("GENERATE_IMAGE good quality image for: " + prompt, local_out_file) 
    global_task_manager.update_task(
                task_id,
                status="Stable Diffusion complete",
                progress=85
            )
    return file_stream, generated_asset_path

async def _run_animatediff(prompt: str, task_id: str | None) -> tuple[io.BytesIO, str]:
    print(f"[GPU Workload] AnimateDiff serving prompt: '{prompt}'")
    local_out_file = f"/server/static/docker_generated_outputs/{task_id}_ai_generated_{prompt}.gif"
    global_task_manager.update_task(
                task_id,
                status="Running Animated Diffusion",
                progress=50
            )
    file_stream, generated_asset_path = background_image_generator.generate_background_gif_webUI("GENERATE_IMAGE good quality animated gif image for: " + prompt, local_out_file)
    global_task_manager.update_task(
                task_id,
                status="Animated Diffusion complete",
                progress=85
            )
    return file_stream, generated_asset_path

async def dispatch_downstream_media(
    json_config: Config,
    config: RunnableConfig,
    task_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, Optional[dict], Optional[str]], None]] = None,
) -> Dict[str, Any]:
    print(f"Dispatching downstream media generation for config: {json_config}")
    task_id = config.get("configurable", {}).get("task_id")
    output_bundle = {
        "config": json_config.model_dump(),
        "generated_asset_path": None,
        "generated_asset_public_url": None
    }

    if task_id:
        global_task_manager.update_task(task_id, status="Invoking media generation model", progress=45)

    # if progress_callback:
    #     try:
    #         progress_callback("LLM complete", 60, None, None)
    #     except Exception:
    #         pass

    if json_config.MediaRequired == "STATIC_PNG" and json_config.MediaPrompt:
        # if progress_callback:
        #     progress_callback("Running Stable Diffusion", 75, None, None)
        file_stream, generated_asset_path = await _run_stable_diffusion(json_config.MediaPrompt, task_id=task_id)
        file_basename = os.path.basename(generated_asset_path)
        storage_service.upload_file_stream(
            file_data=file_stream,
            object_name=f"image/{file_basename}",
            content_type="image/png"
        )

        download_url = storage_service.get_presigned_url(f"image/{file_basename}")

        Config_json = json_config.model_dump_json(indent=2)
        Config_data = json.loads(Config_json)

        # static_path = generated_asset_path.split("static/images/", 1)[1] # file_basename
        # Config_data["Config"]["Background"]["image"] = f"{os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')}/static/images/{static_path}"
        
        Config_data["Config"]["Background"]["image"] = download_url
        Config_data["Config"]["Background"]["color"] = "null"
        Config_json = json.dumps(Config_data, indent=2)


        output_bundle["config"] = Config_json
        output_bundle["generated_asset_path"] = generated_asset_path
        output_bundle["generated_asset_public_url"] = download_url
        # if progress_callback:
        #     progress_callback("Stable Diffusion complete", 90, output_bundle, None)

    elif json_config.MediaRequired == "ANIMATED_GIF" and json_config.MediaPrompt:
        # if progress_callback:
        #     progress_callback("Running Animated Diffusion", 75, None, None)
        file_stream, generated_asset_path = await _run_animatediff(json_config.MediaPrompt, task_id=task_id)
        file_basename = os.path.basename(generated_asset_path)
        storage_service.upload_file_stream(
            file_data=file_stream,
            object_name=f"gif/{file_basename}",
            content_type="image/gif"
        )

        download_url = storage_service.get_presigned_url(f"gif/{file_basename}")

        Config_json = json_config.model_dump_json(indent=2)
        Config_data = json.loads(Config_json)

        # static_path = generated_asset_path.split("static/gifs/", 1)[1] # file_basename
        # Config_data["Config"]["Background"]["image"] = f"{os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')}/static/gifs/{static_path}"

        Config_data["Config"]["Background"]["image"] = download_url
        Config_data["Config"]["Background"]["color"] = "null"
        Config_json = json.dumps(Config_data, indent=2)

        output_bundle["config"] = Config_json
        output_bundle["generated_asset_path"] = generated_asset_path
        output_bundle["generated_asset_public_url"] = download_url
        # if progress_callback:
        #     progress_callback("Animated Diffusion complete", 90, output_bundle, None)
    
    global_task_manager.update_task(task_id, status="LangChain complete", progress=90)
    return output_bundle

def get_orchestration_chain(
    task_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, Optional[dict], Optional[str]], None]] = None,
):
    base_prompt = _load_prompt()

    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_k_m"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.5")),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
    )

    structured_llm = llm.with_structured_output(Config)

    system_prompt = f"{SYSTEM_PROMPT}\n\n{base_prompt}\n\nUse the contextual knowledge provided by the RAG data lake to accurately calibrate JSON schemas."
    system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "User Request: {user_input}\n\nRAG Context:\n{context}")
    ])

    # async def _media_step(config: Config):
    #     if progress_callback:
    #         try:
    #             progress_callback("LLM complete", 60, None, None)
    #         except Exception:
    #             pass
    #     return await dispatch_downstream_media(config, task_id=task_id, progress_callback=progress_callback)

    return prompt_template | structured_llm | RunnableLambda(dispatch_downstream_media)

pipeline_chain = get_orchestration_chain()

async def execute_async_pipeline_job(task_id: str, payload: Dict[str, Any]):
    user_input = payload["user_input"]
    use_rag = payload["use_rag"]
    
    start_time = time.time()
    
    error_msg = None
    final_json = None

    try:
        global_task_manager.update_task(task_id, status="PROCESSING", progress=5)
        context = "No additional context provided."
        docs = []
        if use_rag:
            global_task_manager.update_task(task_id, status="Retrieving RAG context", progress=10)
            docs = query_rag(user_input, k=4)
            context = "\n\n".join(doc["page_content"] for doc in docs)
            global_task_manager.update_task(task_id, status="RAG ready", progress=30)
            
        global_task_manager.update_task(task_id, status="Waiting for available GPU slot", progress=30)
        async with global_task_manager.vram_semaphore:
            print(f"[VRAM Lock Acquired] Processing task {task_id}")
            global_task_manager.update_task(task_id, status="Invoking model", progress=35)
            pipeline_output = await pipeline_chain.ainvoke({
                "user_input": user_input,
                "context": context
            },
            config={"configurable": {"task_id": task_id}})
            
        global_task_manager.update_task(task_id, status="Finalizing output", progress=95)
        final_json = json.dumps(pipeline_output.get("config"))
        global_task_manager.update_task(
            task_id, 
            status="Completed", 
            progress=100, 
            result={
                "config": pipeline_output["config"],
                "generated_asset_path": pipeline_output["generated_asset_path"],
                "rag": {"enabled": use_rag, "documents": docs}
            }
        )
        print(f"[VRAM Lock Released] Task {task_id} compiled successfully.")
            
    except Exception as err:
        error_msg = str(err)
        print(f"Pipeline Execution Failure on task {task_id}: {err}")
        global_task_manager.update_task(
            task_id, 
            status="FAILED", 
            progress=100, 
            error=str(err)
        )
    finally:
        total_latency_ms = (time.time() - start_time) * 1000
        log_telemetry(
            task_id=task_id,
            prompt=user_input,
            status=global_task_manager.get_task_status(task_id).status,
            latency=total_latency_ms,
            rag=use_rag,
            json_out=final_json,
            error=error_msg
        )

def generate_config(user_input: str, context: Optional[str] = None) -> dict:
    if not user_input or not user_input.strip():
        raise ValueError("user_input is required")

    base_prompt = _load_prompt()
    parser = PydanticOutputParser(pydantic_object=Config)
    prompt_template = _build_prompt_template(parser)

    prompt_values = prompt_template.format_prompt(
        base_prompt=base_prompt,
        format_instructions=parser.get_format_instructions(),
        user_input=user_input.strip(),
        context=context or "No additional context provided.",
    )

    llm = _create_ollama_llm()
    result = llm.generate_prompt([prompt_values])

    output = ""
    if result.generations:
        first_generation = result.generations[0][0]
        output = getattr(first_generation, "text", None) or ""
        if not output and hasattr(first_generation, "message"):
            output = getattr(first_generation.message, "content", "")

    try:
        parsed = parser.parse(output)
        return parsed.model_dump()
    except Exception as exc:
        raise ValueError(
            f"Failed to parse config JSON from Ollama response: {exc}\nResponse: {output}"
        )


def generate_config_direct(user_input: str, context: Optional[str] = None) -> dict:
    if not user_input or not user_input.strip():
        raise ValueError("user_input is required")

    base_prompt = _load_prompt()
    prompt_parts = [
        base_prompt,
        "Use the schema exactly and return only valid JSON, no extra text.",
    ]
    if context:
        prompt_parts.append("Use this context to improve reliability:\n" + context)
    prompt_parts.append(f"User request: {user_input.strip()}")
    prompt = "\n\n".join(prompt_parts)

    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_k_m")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.5"))
    response = ollama_chat(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        options={"temperature": temperature},
        format=Config.model_json_schema(),
    )
    response_text = response.message.content if hasattr(response, "message") else response.get("message", {}).get("content", "")
    config = Config.model_validate_json(response_text)
    return config.model_dump()

