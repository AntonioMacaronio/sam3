"""
SAM3 Client

Use this on the remote machine to send images to the SAM3 server and receive masks.

Example usage:
    from sam3_client import SAM3Client

    client = SAM3Client("http://YOUR_SERVER_IP:8000")
    masks = client.predict("path/to/image.jpg", "person")
    print(f"Got {len(masks)} masks with shape {masks.shape}")
"""

import base64
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import requests


class SAM3Client:
    def __init__(self, server_url: str, timeout: int = 120):
        """
        Initialize SAM3 client.

        Args:
            server_url: URL of the SAM3 server (e.g., "http://192.168.1.100:8000")
            timeout: Request timeout in seconds (default 120 for large images)
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        prompt: str,
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Send an image and text prompt to SAM3 server, get masks back.

        Args:
            image: Image path, bytes, or numpy array (RGB)
            prompt: Text prompt describing what to segment
            confidence_threshold: Confidence threshold for detection (default 0.5)

        Returns:
            numpy array of masks with shape (N, 1, H, W) where N is number of masks
        """
        # Convert image to base64
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image, np.ndarray):
            from PIL import Image
            import io
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        else:
            image_bytes = image

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Send request
        response = requests.post(
            f"{self.server_url}/predict",
            json={
                "image_base64": image_base64,
                "prompt": prompt,
                "confidence_threshold": confidence_threshold
            },u
            timeout=self.timeout
        )
        response.raise_for_status()

        # Decode response
        data = response.json()
        masks_bytes = base64.b64decode(data["masks_base64"])
        masks = pickle.loads(masks_bytes)

        return masks

    def health_check(self) -> dict:
        """Check if the server is healthy and model is loaded."""
        response = requests.get(f"{self.server_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Client")
    parser.add_argument("--server", required=True, help="Server URL (e.g., http://192.168.1.100:8000)")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", required=True, help="Text prompt for segmentation")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", help="Output path for masks (numpy .npy file)")

    args = parser.parse_args()

    client = SAM3Client(args.server)

    # Check server health
    print(f"Checking server health at {args.server}...")
    health = client.health_check()
    print(f"Server status: {health}")

    if not health.get("model_loaded"):
        print("Warning: Model not loaded yet, request may take longer...")

    # Run prediction
    print(f"Sending image: {args.image}")
    print(f"Prompt: {args.prompt}")

    masks = client.predict(args.image, args.prompt, args.confidence)

    print(f"Received {len(masks) if masks.size > 0 else 0} masks")
    if masks.size > 0:
        print(f"Mask shape: {masks.shape}")

    # Save if output specified
    if args.output:
        np.save(args.output, masks)
        print(f"Masks saved to: {args.output}")
