"""
SAM3 Client - Send images to remote SAM3 server

Usage:
    python sam3_client.py --url <NGROK_URL> --image <IMAGE_PATH> --prompt <TEXT_PROMPT>

Example:
    python sam3_client.py --url https://abc123.ngrok.io --image photo.jpg --prompt "person"

Requirements:
    pip install requests pillow numpy matplotlib
"""

import argparse
import base64
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image


def send_image_for_prediction(
    server_url: str,
    image_path: str,
    text_prompt: str,
    confidence_threshold: float = 0.5
) -> dict:
    """
    Send an image to the SAM3 server for segmentation.

    Args:
        server_url: The ngrok URL of the SAM3 server (e.g., "https://abc123.ngrok.io")
        image_path: Path to the image file
        text_prompt: Text description of what to segment
        confidence_threshold: Detection confidence threshold (0.0-1.0)

    Returns:
        Dictionary with masks, boxes, scores, and labels
    """
    endpoint = f"{server_url.rstrip('/')}/predict"

    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {
            'text_prompt': text_prompt,
            'confidence_threshold': confidence_threshold
        }
        response = requests.post(endpoint, files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"Server error: {response.status_code} - {response.text}")

    return response.json()


def decode_masks(result: dict) -> list:
    """Decode base64-encoded masks to numpy arrays."""
    masks = []
    for mask_b64 in result.get('masks', []):
        mask_bytes = base64.b64decode(mask_b64)
        mask_img = Image.open(io.BytesIO(mask_bytes))
        masks.append(np.array(mask_img))
    return masks


def visualize_results(image_path: str, result: dict, save_path: str = None):
    """Visualize the segmentation results."""
    image = Image.open(image_path)
    masks = decode_masks(result)

    n_masks = len(masks)
    if n_masks == 0:
        print("No detections found!")
        plt.imshow(image)
        plt.title(f"No detections for prompt: '{result['prompt']}'")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        return

    # Create subplot grid
    cols = min(3, n_masks + 1)
    rows = (n_masks + 1 + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Show original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original\nPrompt: '{result['prompt']}'")
    axes[0].axis('off')

    # Show each mask
    scores = result.get('scores', [])
    labels = result.get('labels', [])

    for i, mask in enumerate(masks):
        ax = axes[i + 1]

        # Overlay mask on image
        img_array = np.array(image)
        overlay = img_array.copy()
        mask_bool = mask > 127

        # Apply colored mask overlay
        color = plt.cm.tab10(i % 10)[:3]
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                overlay[:, :, c] * 0.5 + int(color[c] * 255) * 0.5,
                overlay[:, :, c]
            )

        ax.imshow(overlay.astype(np.uint8))

        title = f"Mask {i + 1}"
        if i < len(scores):
            title += f"\nScore: {scores[i]:.3f}"
        if i < len(labels):
            title += f"\nLabel: {labels[i]}"
        ax.set_title(title)
        ax.axis('off')

    # Hide unused subplots
    for i in range(n_masks + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Results saved to: {save_path}")
    plt.show()


def check_server_health(server_url: str) -> bool:
    """Check if the server is ready."""
    try:
        response = requests.get(f"{server_url.rstrip('/')}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('model_loaded', False)
    except requests.exceptions.RequestException:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description='SAM3 Client')
    parser.add_argument('--url', required=True, help='Server URL (e.g., https://abc123.ngrok.io)')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--prompt', required=True, help='Text prompt for segmentation')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--output', help='Path to save output visualization')

    args = parser.parse_args()

    # Check server health
    print(f"Checking server at {args.url}...")
    if not check_server_health(args.url):
        print("Warning: Server may not be ready or unreachable")

    # Send image for prediction
    print(f"Sending image: {args.image}")
    print(f"Text prompt: {args.prompt}")

    result = send_image_for_prediction(
        server_url=args.url,
        image_path=args.image,
        text_prompt=args.prompt,
        confidence_threshold=args.confidence
    )

    print(f"\nResults:")
    print(f"  Detections: {result['num_detections']}")
    print(f"  Image size: {result['image_size']}")

    if result['num_detections'] > 0:
        print(f"  Scores: {result['scores']}")
        print(f"  Labels: {result['labels']}")

    # Visualize results
    visualize_results(args.image, result, args.output)


if __name__ == "__main__":
    main()