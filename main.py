import os
import base64
import random
import logging
from io import BytesIO
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
from dotenv import load_dotenv
from diffusers import StableDiffusionUpscalePipeline

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

# Ensuring the directory
folder_path = "./static/images"
os.makedirs(folder_path, exist_ok=True)
logger.info(f"Ensured {folder_path} directory exists.")

# Load model and scheduler
model_id = os.getenv("MODEL_NAME")
if not model_id:
    logger.error("MODEL_NAME environment variable is not set.")
    raise ValueError("MODEL_NAME environment variable is not set.")

logger.info(f"Loading model: {model_id}")

try:
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    )
    pipeline = pipeline.to("cuda")
    logger.info("Model loaded successfully and moved to CUDA.")
except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    raise ValueError("Model loading failed.")


@app.post("/api/inpaint/upscale/")
async def inpaint_image(
    file: UploadFile = File(...),
    target_width: int = 512,
    target_height: int = 512,
    prompt: Optional[str] = "",
    is_base64: bool = False
):
    """
    Inpaints and upscales an image based on the given parameters.

    Parameters:
    ----------
    `file`: UploadFile
        The image file to be processed, uploaded by the user.
    `target_width`: int, optional
        The target width of the image after processing, default is 512 pixels.
        Maximum allowed width is 1024 pixels.
    `target_height`: int, optional
        The target height of the image after processing, default is 512 pixels.
        Maximum allowed height is 1024 pixels.
    `prompt`: str, optional
        An optional text prompt to guide the inpainting process.
        Default is an empty string.
    `is_base64`: bool, optional
        If set to True, the processed image will be returned in base64 format.
        Otherwise, a file response will be returned. Default is False.

    Returns:
    -------
    JSONResponse or FileResponse
        - If is_base64 is False, a FileResponse containing the upscaled image
          will be returned.
        - If is_base64 is True, a JSONResponse containing the base64-encoded
          version of the upscaled image will be returned.
    """
    try:
        if target_height > 1024 or target_width > 1024:
            return JSONResponse(
                content=(
                    "The image size exceeds the allowed limit of 1024px "
                    "in either width or height."
                ),
                status_code=status.HTTP_400_BAD_REQUEST
            )

        logger.info("Received request to inpaint image.")
        low_res_img = Image.open(BytesIO(await file.read())).convert("RGB")
        logger.info("Image loaded successfully.")

        # Rescale the image
        scale_factor = 4
        target_width = max(target_width // scale_factor, 1)
        target_height = max(target_height // scale_factor, 1)
        low_res_img = low_res_img.resize((target_width, target_height))
        logger.info(f"Image resized to {target_width}x{target_height}.")

        # Upscale the image
        random_num = random.randint(0, 10000)
        upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]

        file_name = "".join(file.filename.split('.')[:-1])
        output_path = f"{folder_path}/upscaled_{random_num}_{file_name}.png"
        upscaled_image.save(output_path)
        logger.info(f"Image upscaled and saved to {output_path}.")

        if not is_base64:
            return FileResponse(output_path)

        # Convert the upscaled image to base64
        buffered = BytesIO()

        # Saving the image to the buffer in PNG format
        upscaled_image.save(buffered, format="PNG")
        buffered.seek(0)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info("Image converted to base64 successfully.")
        return JSONResponse(
            {"image": image_base64},
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image processing failed.")


@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Image Inpainting API is running!"}
