from diffusers import StableDiffusionPipeline, AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
import torch

    
model_id = "stabilityai/stable-diffusion-2-0-base"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,
    revision="fp16"
).to("cuda")

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16)
gif_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
gif_pipe = AnimateDiffPipeline.from_pretrained(
    gif_model_id,
    torch_dtype=torch.float16,
    motion_adapter=adapter
).to("cuda")
    
scheduler = DDIMScheduler.from_pretrained(
    gif_model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
gif_pipe.scheduler = scheduler
    
gif_pipe.enable_vae_slicing()
# gif_pipe.enable_model_cpu_offload()
    
def generate_background_image(word: str, output_path: str):

    prompt_text = word

    image = pipe(
        prompt_text,
        num_inference_steps=30,
        guidance_scale=8.0 ,
        width=360,
        height=360,
    ).images[0]

    image.save(output_path)
    print("Saved to", output_path)
    return output_path

def generate_background_gif(word: str, output_path: str):

    gif = gif_pipe(
        word,
        negative_prompt="best quality, good quality",
        num_frames=16,
        num_inference_steps=30,
        guidance_scale=8.0 ,
        width=360,
        height=360,
    ).frames[0]
    export_to_gif(gif, output_path, fps=10)
    print("Saved to", output_path)
    return output_path