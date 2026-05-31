import os
import json
from ollama import chat
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from server.try_rag_storage import initialize_example_selector
from server.schemas import Config

example_selector = initialize_example_selector()

SYSTEM_PROMPT = """You are an expert wearable device UI/UX configuration engine. 
Your sole task is to generate valid architectural JSON layout models for smartwatches based on user text queries.
You must absolutely conform your reasoning to match the format of the examples provided below.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", "Contextual Example:\nInput: {example_input}\nOutput: {example_output}"),
    ("user", "Target Request: {user_prompt}")
])

# raw_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
response = chat(
      messages=[
        {
          'role': 'user',
          'content': input,
        }
      ],
      options={'temperature': 0.5},
      model='llama3.1:8b',
      format=Config.model_json_schema(),
    )
structured_llm = response.with_structured_output(Config)

generation_chain = prompt_template | structured_llm

async def generate_config_json(user_input: str) -> str:
    best_match = example_selector.select_examples({"input": user_input})[0]
    
    config_object: Config = await generation_chain.ainvoke({
        "user_prompt": user_input,
        "example_input": best_match["input"],
        "example_output": best_match["output"]
    })
    
    return config_object.model_dump_json()