from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Callable

from chromadb import config
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
from server.background_image_generate import generate_background_gif, generate_background_image
from server.schemas import Config
from server.task_manager import global_task_manager
from server.rag_vector_store import query_rag

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

SYSTEM_PROMPT = """You are an expert wearable device UI/UX configuration engine. 
Your sole task is to generate valid architectural JSON layout models for smartwatches based on user text queries.
You must absolutely conform your reasoning to match the format of the examples provided below.
"""

def _load_prompt() -> str:
    prompt_path = ROOT_DIR / "datasets" / "prompts.txt"
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
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.5")),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

async def _run_stable_diffusion(prompt: str, task_id: str | None) -> str:
    print(f"[GPU Workload] Stable Diffusion serving prompt: '{prompt}'")
    out_file = f"/server/static/images/ai_generated_{prompt}.png"
    global_task_manager.update_task(
                task_id,
                status="Running Stable Diffusion",
                progress=50
            )
    generated_asset_path = generate_background_image("GENERATE_IMAGE good quality image for: " + prompt, out_file) 
    global_task_manager.update_task(
                task_id,
                status="Stable Diffusion complete",
                progress=85
            )
    return generated_asset_path

async def _run_animatediff(prompt: str, task_id: str | None) -> str:
    print(f"[GPU Workload] AnimateDiff serving prompt: '{prompt}'")
    out_file = f"/server/static/gifs/ai_generated_{prompt}.gif"
    global_task_manager.update_task(
                task_id,
                status="Running Animated Diffusion",
                progress=50
            )
    generated_asset_path = generate_background_gif("GENERATE_IMAGE good quality animated gif image for: " + prompt, out_file)
    global_task_manager.update_task(
                task_id,
                status="Animated Diffusion complete",
                progress=85
            )
    return generated_asset_path

async def dispatch_downstream_media(
    config: Config,
    runnableConfig: RunnableConfig,
    task_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, Optional[dict], Optional[str]], None]] = None,
) -> Dict[str, Any]:

    task_id = runnableConfig.get("configurable", {}).get("task_id")
    output_bundle = {
        "config": config.model_dump(),
        "generated_asset_path": None,
    }

    if task_id:
        global_task_manager.update_task(task_id, status="Invoking media generation model", progress=45)

    # if progress_callback:
    #     try:
    #         progress_callback("LLM complete", 60, None, None)
    #     except Exception:
    #         pass

    if config.media_required == "STATIC_PNG" and config.media_prompt:
        # if progress_callback:
        #     progress_callback("Running Stable Diffusion", 75, None, None)
        generated_asset_path = await _run_stable_diffusion(config.media_prompt, task_id=task_id)
        Config_json = Config.model_dump_json(indent=2)
        Config_data = json.loads(Config_json)
        static_path = generated_asset_path.split("static/images/", 1)[1]
        Config_data["Config"]["Background"]["image"] = f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/static/images/{static_path}"
        Config_data["Config"]["Background"]["color"] = "null"
        Config_json = json.dumps(Config_data, indent=2)
        output_bundle["config"] = Config_json
        output_bundle["generated_asset_path"] = generated_asset_path
        # if progress_callback:
        #     progress_callback("Stable Diffusion complete", 90, output_bundle, None)

    elif config.media_required == "ANIMATED_GIF" and config.media_prompt:
        # if progress_callback:
        #     progress_callback("Running Animated Diffusion", 75, None, None)
        generated_asset_path = await _run_animatediff(config.media_prompt, task_id=task_id)
        Config_json = Config.model_dump_json(indent=2)
        Config_data = json.loads(Config_json)
        static_path = generated_asset_path.split("static/gifs/", 1)[1]
        Config_data["Config"]["Background"]["image"] = f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/static/gifs/{static_path}"
        Config_data["Config"]["Background"]["color"] = "null"
        Config_json = json.dumps(Config_data, indent=2)
        output_bundle["config"] = Config_json
        output_bundle["generated_asset_path"] = generated_asset_path
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
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.5")),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    structured_llm = llm.with_structured_output(Config)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"{SYSTEM_PROMPT}\n\n{base_prompt}\n\nUse the contextual knowledge provided by the RAG data lake to accurately calibrate JSON schemas."),
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
        print(f"Pipeline Execution Failure on task {task_id}: {err}")
        global_task_manager.update_task(
            task_id, 
            status="FAILED", 
            progress=100, 
            error=str(err)
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

    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
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

