from diffusers import StableDiffusionPipeline, AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
import torch


adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        motion_adapter=adapter
    ).to("cuda")
    
scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
pipe.scheduler = scheduler
    
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()


gif = pipe(
        "thanos",
        negative_prompt="best quality, good quality",
        num_frames=16,
        num_inference_steps=30,
        guidance_scale=8.0 ,
        width=512,
        height=512,
    ).frames[0]
output_path = f"/static/images/ai_generated_thanos.gif"
export_to_gif(gif, output_path, fps=10)
print("Saved to", output_path)