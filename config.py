from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaTokenizer, LlamaForCausalLM
import torch
import json
import os
from diffusers import StableDiffusionPipeline

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  
)

llama_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

with open('.\datasets\prompts.txt', 'r', encoding='utf-8') as f:
    base_prompt = f.read()

def generate_background_image(word: str, output_path: str):

    model_id = "stabilityai/stable-diffusion-xl-base-1.0" 
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        revision="fp16"
    ).to("cuda")

    prompt_text = word

    image = pipe(
        prompt_text,
        num_inference_steps=30, 
        guidance_scale=8.0 ,
        width=512,
        height=512,
    ).images[0]

    image.save(output_path)
    print("Saved to", output_path)
    return output_path

def extract_complete_json(raw_output: str) -> str:

    raw = raw_output.strip()
    start = raw.find("{")
    if start == -1:
        raise RuntimeError(f"No opening brace found in model output:\n{raw_output}")

    depth = 0
    for idx in range(start, len(raw)):
        if raw[idx] == "{":
            depth += 1
        elif raw[idx] == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]

    raise RuntimeError(f"Could not find matching closing brace in model output:\n{raw_output}")


def build_json_config(user_text: str, host: str) -> dict:

    prompt_input = base_prompt + f"\n\nNow generate a valid config.json for:\n“{user_text}”\n"
    output = llama_pipe(
        prompt_input,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.5,
        top_p=0.9
    )[0]["generated_text"]

    raw = output.replace(prompt_input, "").strip()
    try:
        json_only = extract_complete_json(raw)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to extract a complete JSON block:\n{raw}") from exc

    try:
        config_skeleton = json.loads(json_only)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from model:\n{json_only}") from e
    # generated_json_str = output.replace(prompt_input, "").strip()

    # start = generated_json_str.find("{")
    # end   = generated_json_str.rfind("}")


    # if start == -1 or end == -1 or end < start:
    #     raise RuntimeError(f"Could not locate JSON braces in model output:\n{generated_json_str}")
    
    # json_only = generated_json_str[start : end+1]


    # try:
    #     config_skeleton = json.loads(json_only)
    # except json.JSONDecodeError as e:
    #     raise RuntimeError(f"Failed to parse JSON from model:\n{json_only}") from e

    bg = config_skeleton["Config"]["Background"]
    if bg.get("image") and bg["image"].startswith("/image/path/to/"):
        placeholder = bg["image"]
        img_prompt = placeholder.split("/")[-1][len("ai_generated_"):-len(".png")]
        out_file = f".\static\bg_{img_prompt}.png"
        local_path = generate_background_image("GENERATE_IMAGE good quality image for: " + img_prompt, out_file)
        bg["image"] = f"{host}/static/bg_{img_prompt}.png"
        bg["color"] = None
    return config_skeleton