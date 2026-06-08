import base64

from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, AnimateDiffPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from dotenv import load_dotenv
import torch
import os
import io
import requests


class BackgroundImageGenerator:
    def __init__(self):
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        self.url = "http://host.docker.internal:7860/sdapi/v1/txt2img"

    def _init_image_generation_pipeline(self):
        model_id = "OFA-Sys/small-stable-diffusion-v0/" # "sd-legacy/stable-diffusion-v1-5" # "stabilityai/stable-diffusion-3.5-large-turbo"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            token=self.hf_token,
            use_auth_token=self.hf_token,
        )
        pipe.enable_model_cpu_offload()
        return pipe
    
    def _init_gif_generation_pipeline(self):
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-3",
            torch_dtype=torch.float16,
            token=self.hf_token,
            use_auth_token=self.hf_token,
        )

        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        pipe = AnimateDiffPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            motion_adapter=adapter,
            token=self.hf_token,
            use_auth_token=self.hf_token,
        )
        
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
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
        return pipe
    
    def generate_background_image_webUI(self, word: str, output_path: str):
        prompt_text = word

        payload = {
            "prompt": f"{prompt_text}, photorealistic, highly detailed, 8k resolution",
            "negative_prompt": "blurry, low quality, distorted, deformed, bad anatomy, cartoon, drawing, painting, illustration, text, watermark, bad proportions",
            "steps": 25,
            "cfg_scale": 7.0,
            "width": 512,
            "height": 512,
            "sampler_name": "DPM++ 2M Karras" 
        }

        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            result = response.json()
            image_base64 = result['images'][0]
            image_data = base64.b64decode(image_base64) # AUTOMATIC1111 does not return raw image data. It returns a JSON text object containing metadata and an array of images encoded as long Base64 strings
            docker_output_path = os.path.join("/app/server/static/docker_generated_outputs", os.path.basename(output_path)) 
    
            with open(docker_output_path, "wb") as f:
                f.write(image_data)
            print(f"Saved to local {output_path}")
            return io.BytesIO(image_data), output_path
        except Exception as e:
            print(f"Error connecting to AUTOMATIC1111 API: {e}")
            raise e
        
    def generate_background_image(self, word: str, output_path: str):

        prompt_text = word

        image = self.pipe(
            prompt_text,
            num_inference_steps=4,
            guidance_scale=0.0 ,
            width=360,
            height=360,
        ).images[0]
        file_stream = io.BytesIO()

        image.save(file_stream, format="PNG")
        file_stream.seek(0)

        image.save(output_path)
        print(f"Saved to local {output_path}")
        return file_stream, output_path
    
    def generate_background_gif(self, word: str, output_path: str):

        prompt_text = word

        output = self.gif_pipe(
            prompt_text,
            negative_prompt="bad quality, blurry, deformed",
            num_frames=16,
            num_inference_steps=30,
            guidance_scale=8.0,
            generator=torch.Generator("cpu").manual_seed(42),
            width=360,
            height=360,
        )
        gif_frames = output.frames[0]

        file_stream = io.BytesIO()
        frame_duration = int(1000 / 10)  # 10 FPS

        gif_frames[0].save(
            file_stream,
            format="GIF",
            save_all=True,
            append_images=gif_frames[1:],
            duration=frame_duration,
            loop=0
        )
        file_stream.seek(0)

        export_to_gif(gif_frames, output_path, fps=10)
        print(f"Saved to local {output_path}")
        return file_stream, output_path 

    def generate_background_gif_webUI(self, word: str, output_path: str):

        payload = {
            "prompt": f"{word}, masterpiece, cinematic motion, high quality",
            "negative_prompt": "static, still frame, deformed, bad quality, blurry",
            "steps": 20,
            "cfg_scale": 7.5,
            "width": 512,
            "height": 512,
            "sampler_name": "Euler a",
            
            "alwayson_scripts": {
                "AnimateDiff": {
                    "args": [{
                        "enable": True,
                        "model": "diffusion_pytorch_model.fp16.safetensors",
                        "video_length": 16,
                        "fps": 8,
                        "format": ["GIF"],
                        "loop_number": 0
                    }]
                }
            }
        }

        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        image_base64 = result['images'][0]
        gif_data = base64.b64decode(image_base64)

        docker_output_path = os.path.join("/app/server/static/docker_generated_outputs", os.path.basename(output_path)) 
    
        with open(docker_output_path, "wb") as f:
            f.write(gif_data)
        print(f"Saved to local {output_path}")
        return io.BytesIO(gif_data), output_path
    
background_image_generator = BackgroundImageGenerator()