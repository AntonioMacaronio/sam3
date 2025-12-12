"""
SAM3 Image Prediction Server with ngrok

Run with:
    python sam3_server.py

The server will automatically create an ngrok tunnel and print the public URL.
Share that URL with the client machine.

Requirements:
    pip install fastapi uvicorn pyngrok python-multipart

First time setup for ngrok:
    1. Create account at https://ngrok.com
    2. Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
    3. Run: ngrok config add-authtoken YOUR_TOKEN
"""

import base64
import io
import os
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pyngrok import ngrok

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Enable TF32 for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

app = FastAPI(title="SAM3 Image Prediction Server")

# Global model and processor (loaded on startup)
model = None
processor = None


def load_model():
    """Load the SAM3 model."""
    global model, processor
    print("Loading SAM3 model...")
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    print("SAM3 model loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        load_model()


def encode_mask_to_base64(mask: torch.Tensor) -> str:
    """Convert a mask tensor to a base64-encoded PNG image."""
    mask_np = mask.squeeze().cpu().numpy()
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode='L')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Predict segmentation masks for an image using a text prompt.

    Args:
        image: Image file to segment
        text_prompt: Text description of what to segment (e.g., "person", "car")
        confidence_threshold: Confidence threshold for detections (0.0-1.0)

    Returns:
        JSON with masks (base64-encoded PNGs), boxes, scores, and labels
    """
    global processor

    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        width, height = pil_image.size

        if processor.confidence_threshold != confidence_threshold:
            processor.confidence_threshold = confidence_threshold

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inference_state = processor.set_image(pil_image)
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt
            )

        masks = inference_state.get("masks", torch.tensor([]))
        boxes = inference_state.get("boxes", torch.tensor([]))
        scores = inference_state.get("scores", torch.tensor([]))
        labels = inference_state.get("labels", [])

        encoded_masks = []
        if len(masks) > 0:
            for mask in masks:
                encoded_masks.append(encode_mask_to_base64(mask))

        boxes_list = boxes.cpu().tolist() if len(boxes) > 0 else []
        scores_list = scores.cpu().tolist() if len(scores) > 0 else []

        return JSONResponse({
            "success": True,
            "num_detections": len(encoded_masks),
            "image_size": {"width": width, "height": height},
            "masks": encoded_masks,
            "boxes": boxes_list,
            "scores": scores_list,
            "labels": labels,
            "prompt": text_prompt
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_base64")
async def predict_base64(
    image_base64: str = Form(...),
    text_prompt: str = Form(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Alternative endpoint accepting base64-encoded image.
    """
    global processor

    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        pil_image = decode_base64_to_image(image_base64)
        width, height = pil_image.size

        if processor.confidence_threshold != confidence_threshold:
            processor.confidence_threshold = confidence_threshold

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inference_state = processor.set_image(pil_image)
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt
            )

        masks = inference_state.get("masks", torch.tensor([]))
        boxes = inference_state.get("boxes", torch.tensor([]))
        scores = inference_state.get("scores", torch.tensor([]))
        labels = inference_state.get("labels", [])

        encoded_masks = []
        if len(masks) > 0:
            for mask in masks:
                encoded_masks.append(encode_mask_to_base64(mask))

        boxes_list = boxes.cpu().tolist() if len(boxes) > 0 else []
        scores_list = scores.cpu().tolist() if len(scores) > 0 else []

        return JSONResponse({
            "success": True,
            "num_detections": len(encoded_masks),
            "image_size": {"width": width, "height": height},
            "masks": encoded_masks,
            "boxes": boxes_list,
            "scores": scores_list,
            "labels": labels,
            "prompt": text_prompt
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check if the server and model are ready."""
    return {
        "status": "healthy" if processor is not None else "loading",
        "model_loaded": processor is not None
    }


def start_ngrok(port: int) -> str:
    """Start ngrok tunnel and return the public URL."""
    public_url = ngrok.connect(port)
    print("\n" + "=" * 60)
    print("NGROK TUNNEL ACTIVE")
    print("=" * 60)
    print(f"Public URL: {public_url}")
    print(f"Share this URL with the client machine!")
    print("=" * 60 + "\n")
    return public_url


if __name__ == "__main__":
    PORT = 8000

    # Start ngrok tunnel
    public_url = start_ngrok(PORT)

    # Run the FastAPI server
    print(f"Starting SAM3 server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)