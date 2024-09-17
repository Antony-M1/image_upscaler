import os
import base64
import uuid
import random
import logging
from io import BytesIO
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
from dotenv import load_dotenv
from diffusers import StableDiffusionUpscalePipeline, AutoPipelineForInpainting

load_dotenv()

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

folder_path = "./static/images"
os.makedirs(folder_path, exist_ok=True)
logger.info(f"Ensured {folder_path} directory exists.")

inpainting_model_id = os.getenv("INPAINTING_MODEL_NAME", "stabilityai/stable-diffusion-2-inpainting")
upscale_model_id = os.getenv("UPSCALE_MODEL_NAME", "stabilityai/stable-diffusion-x4-upscaler")

logger.info(f"Loading models: Inpainting - {inpainting_model_id}, Upscale - {upscale_model_id}")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

try:
    upscale_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        upscale_model_id, revision="fp16", torch_dtype=torch.float16
    ).to(device)
    
    inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
        inpainting_model_id, revision="fp16", torch_dtype=torch.float16
    ).to(device)
    
    logger.info("Both models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {e}", exc_info=True)
    raise ValueError("Model loading failed. Please check model files or connection.")


@app.post("/api/upscale/")
async def upscale_image(
    file: UploadFile = File(...),
    target_width: int = 512,
    target_height: int = 512,
    prompt: Optional[str] = "",
    is_base64: bool = False
):
    """
    Upscales an image based on the given parameters.

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
        An optional text prompt to guide the upscaling process.
        Default is an empty string.
    `is_base64`: bool, optional
        If set to True, the processed image will be returned in base64 format.
        Otherwise, a file response will be returned. Default is False.
    """
    try:
        if target_height > 1024 or target_width > 1024:
            return JSONResponse(
                content={"error": "The image size exceeds the allowed limit of 1024px in either width or height."},
                status_code=status.HTTP_400_BAD_REQUEST
            )

        logger.info("Received request to upscale image.")
        low_res_img = Image.open(BytesIO(await file.read())).convert("RGB")
        logger.info("Image loaded successfully.")

        logger.info(f"Upscaling image to {target_width}x{target_height}.")
        random_num = random.randint(0, 10000)
        upscaled_image = upscale_pipeline(prompt=prompt, image=low_res_img).images[0]

        # file_name = "".join(file.filename.split('.')[:-1])
        # output_path = f"{folder_path}/upscaled_{random_num}_{file_name}.png"
        sanitized_filename = f"{uuid.uuid4()}.png"
        output_path = f"{folder_path}/{sanitized_filename}"
        upscaled_image.save(output_path)
        logger.info(f"Image upscaled and saved to {output_path}.")

        if not is_base64:
            return FileResponse(output_path)

        buffered = BytesIO()
        upscaled_image.save(buffered, format="PNG")
        buffered.seek(0)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info("Image converted to base64 successfully.")
        return JSONResponse({"image": image_base64}, status_code=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image processing failed.")


@app.post("/api/inpaint/")
async def inpaint_image(
    file: UploadFile = File(...),
    mask_file: UploadFile = File(...),
    prompt: Optional[str] = "",
    is_base64: bool = False
):
    """
    Inpaint an image using the uploaded mask.

    Parameters:
    ----------
    `file`: UploadFile
        The image file to be processed, uploaded by the user.
    `mask_file`: UploadFile
        The mask image to guide the inpainting process, uploaded by the user.
    `prompt`: str, optional
        An optional text prompt to guide the inpainting process.
    `is_base64`: bool, optional
        If set to True, the processed image will be returned in base64 format.
        Otherwise, a file response will be returned.
    """
    try:
        logger.info("Received request to inpaint image.")
        img = Image.open(BytesIO(await file.read())).convert("RGB")
        mask_img = Image.open(BytesIO(await mask_file.read())).convert("RGB")
        logger.info("Image and mask loaded successfully.")

        random_num = random.randint(0, 10000)
        inpainted_img = inpainting_pipeline(prompt=prompt, image=img, mask_image=mask_img).images[0]

        # file_name = "".join(file.filename.split('.')[:-1])
        # output_path = f"{folder_path}/inpainted_{random_num}_{file_name}.png"
        sanitized_filename = f"{uuid.uuid4()}.png"
        output_path = f"{folder_path}/{sanitized_filename}"
        inpainted_img.save(output_path)
        logger.info(f"Image inpainted and saved to {output_path}.")

        if not is_base64:
            return FileResponse(output_path)

        buffered = BytesIO()
        inpainted_img.save(buffered, format="PNG")
        buffered.seek(0)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info("Image converted to base64 successfully.")
        return JSONResponse({"image": image_base64}, status_code=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image processing failed.")


@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Image Inpainting API is running!"}
