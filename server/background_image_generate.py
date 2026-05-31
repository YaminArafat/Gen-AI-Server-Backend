from diffusers import StableDiffusion3Pipeline, AnimateDiffPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
import torch
import os

hf_token = os.getenv("HF_TOKEN")

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,
    token=hf_token,
    use_auth_token=hf_token,
)
pipe.enable_model_cpu_offload()

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-3",
    torch_dtype=torch.float16,
    token=hf_token,
    use_auth_token=hf_token,
)
gif_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
gif_pipe = AnimateDiffPipeline.from_pretrained(
    gif_model_id,
    torch_dtype=torch.float16,
    motion_adapter=adapter,
    token=hf_token,
    use_auth_token=hf_token,
)
    
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    gif_model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
gif_pipe.scheduler = scheduler
    
gif_pipe.enable_vae_slicing()
gif_pipe.enable_model_cpu_offload()
    
def generate_background_image(word: str, output_path: str):

    prompt_text = word

    image = pipe(
        prompt_text,
        num_inference_steps=4,
        guidance_scale=0.0 ,
        width=360,
        height=360,
    ).images[0]

    image.save(output_path)
    print("Saved to", output_path)
    return output_path

def generate_background_gif(word: str, output_path: str):

    output = gif_pipe(
        word,
        negative_prompt="bad quality, blurry, deformed",
        num_frames=16,
        num_inference_steps=30,
        guidance_scale=8.0,
        generator=torch.Generator("cpu").manual_seed(42),
        width=360,
        height=360,
    )
    gif_frames = output.frames[0]
    export_to_gif(gif_frames, output_path, fps=10)
    print("Saved to", output_path)
    return output_path