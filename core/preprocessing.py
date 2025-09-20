import cv2
from PIL import Image
import io
import numpy as np

def load_image(image_bytes):
    """Load image from bytes for model prediction"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")

def load_image_from_path(image_path):
    """Load image from file path"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image from path")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)
    except Exception as e:
        raise ValueError(f"Error loading image from path: {str(e)}")