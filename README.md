
# Image Inpainting and Upscaling API

This project provides a FastAPI-based service for image upscaling using a pre-trained Stable Diffusion model. It allows users to upload low-resolution images, which are then processed and upscaled using NVIDIA GPU acceleration for optimal performance.

<details>
  <summary><b>Example Output</b></summary>

![image_2024_09_12T14_03_41_124Z](https://github.com/user-attachments/assets/3637a498-3132-4bfa-818d-81aa9c619341)


</details>

[Sample Demo Video](https://drive.google.com/file/d/1ZMYXOMQTGZd9NgJUOpJkIbt7heNhScET/view?usp=sharing)

## Key Features
- Image Upload & Upscaling: Upload an image and get an upscaled version.
- CUDA GPU Support: Leverages local GPUs for faster model inference.
- File or Base64 Response: Choose to receive the result as a file or Base64-encoded image.
- Logging: Logs key operations such as model loading and image processing.

## Project Structure
- `main.py`: FastAPI application handling image uploads and processing.
- `load_model.py`: Script to preload the model and test the load time.
- `Dockerfile`: Configuration for Docker with CUDA and PyTorch.
- `docker-compose.yml`: Manages the app's deployment with GPU support.
- `examples/`: Test images.
- `static/images/`: Stores upscaled images.

## Setup Instructions

### Clone the Repository
Clone the repository and navigate into the project directory:

```cmd
git clone https://github.com/Antony-M1/image_upscaler.git
cd image_upscaler
```

### Docker Setup (Recommended)

#### Prerequisites
- Docker and Docker Compose
- NVIDIA drivers with CUDA support
- Python 3.10+
- Compatible NVIDIA GPU
- Need minimum T4 GPU

#### Environment Variables
Create a `.env` file with your Hugging Face model name:

```env
MODEL_NAME="stabilityai/stable-diffusion-x4-upscaler"
```

#### Build & Run the App (Docker)
Build the Docker image:

```cmd
docker-compose build
```

Start the service:

```cmd
docker-compose up -d
```

Access the API at `http://localhost:8000`.
For **Swagger** `http://localhost:8000/docs`

### Manual Setup (Without Docker)

#### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA and cuDNN installed
- PyTorch with CUDA support
- FastAPI, Uvicorn, and other dependencies from `requirements.txt`

#### Environment Variables
Create a `.env` file in the project root directory:

```env
MODEL_NAME="stabilityai/stable-diffusion-x4-upscaler"
```

#### Steps

1. **Set up a virtual environment**:

   ```cmd
   python3 -m venv venv
   ```

   **Windows Powershell**
   ```
   .\venv\Scripts\activate
   ```

   **Linux Ubuntu**
   ```
   source venv/bin/activate
   ```

2. **Install dependencies**:

   ```cmd
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```

3. **Load the model**:

   Run the `load_model.py` script to ensure the model is loaded correctly:

   ```cmd
   python load_model.py
   ```

4. **Run the FastAPI app**:

   Use Uvicorn to start the server:

   ```cmd
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

Access the API at `http://localhost:8000`.

For **Swagger** `http://localhost:8000/docs`

## API Endpoints
- `GET /`: Health check for the API.
- `POST /api/inpaint/upscale/`: Upload an image and get the upscaled version.
