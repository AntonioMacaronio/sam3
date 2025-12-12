"""
SAM3 Inference Server

Run with: uvicorn sam3_server:app --host 0.0.0.0 --port 8000
Or: python sam3_server.py
"""

import base64
import io
import pickle
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Initialize FastAPI app
app = FastAPI(title="SAM3 Inference Server")

# Global model variables (loaded once at startup)
model = None
processor = None


class PredictRequest(BaseModel):
    image_base64: str  # Base64 encoded image
    prompt: str  # Text prompt for segmentation
    confidence_threshold: Optional[float] = 0.5


class PredictResponse(BaseModel):
    masks_base64: str  # Base64 encoded pickled numpy array
    num_masks: int
    mask_shape: list


@app.on_event("startup")
async def load_model():
    """Load SAM3 model at startup."""
    global model, processor

    print("Loading SAM3 model...")

    # Enable TF32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Build model
    sam3_root = sam3.__path__[0]
    bpe_path = f"{sam3_root}/../assets/bpe_simple_vocab_16e6.txt.gz"

    with torch.autocast("cuda", dtype=torch.bfloat16):
        model = build_sam3_image_model(bpe_path=bpe_path)

    print("SAM3 model loaded successfully!")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Run SAM3 inference on an image with a text prompt."""
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run inference with autocast
        with torch.autocast("cuda", dtype=torch.bfloat16):
            proc = Sam3Processor(model, confidence_threshold=request.confidence_threshold)
            inference_state = proc.set_image(image)
            inference_state = proc.set_text_prompt(
                state=inference_state,
                prompt=request.prompt
            )

        # Extract masks
        masks = inference_state["masks"]

        if masks.numel() == 0:
            # No masks found
            masks_np = np.array([])
            num_masks = 0
            mask_shape = [0]
        else:
            masks_np = masks.cpu().numpy()
            num_masks = masks_np.shape[0]
            mask_shape = list(masks_np.shape)

        # Serialize masks using pickle
        masks_bytes = pickle.dumps(masks_np)
        masks_base64 = base64.b64encode(masks_bytes).decode("utf-8")

        return PredictResponse(
            masks_base64=masks_base64,
            num_masks=num_masks,
            mask_shape=mask_shape
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
