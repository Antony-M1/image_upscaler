import torch
import time
import os
from diffusers import StableDiffusionUpscalePipeline, AutoPipelineForInpainting
from dotenv import load_dotenv


load_dotenv()
inpainting_model_id = os.getenv("INPAINTING_MODEL_NAME", "stabilityai/stable-diffusion-2-inpainting")
upscale_model_id = os.getenv("UPSCALE_MODEL_NAME", "stabilityai/stable-diffusion-x4-upscaler")
try:
    start_time = time.time()
    upscale_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        upscale_model_id, revision="fp16", torch_dtype=torch.float16
    ).to("cuda")
    
    # Load the inpainting pipeline
    inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
        inpainting_model_id, revision="fp16", torch_dtype=torch.float16
    ).to("cuda")
    end_time = time.time()
    print(f"Time taken to load pipeline: {end_time - start_time} seconds")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    print("Failed to load Stable Diffusion pipeline from Hugging Face.")
