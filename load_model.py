import torch
import time
import os
from diffusers import StableDiffusionUpscalePipeline
from dotenv import load_dotenv


load_dotenv()
model_id = os.getenv("MODEL_NAME")
try:
    start_time = time.time()
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    )
    end_time = time.time()
    print(f"Time taken to load pipeline: {end_time - start_time} seconds")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    print("Failed to load Stable Diffusion pipeline from Hugging Face.")
