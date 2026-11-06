
import cv2
import numpy as np
from PIL import Image


def preprocess_canvas(image_array, padding=20):
    """
    Prepare a Streamlit canvas image for AI recognition.

    Args:
        image_array: RGBA image from Streamlit canvas.
        padding: Extra white space around handwriting.

    Returns:
        PIL RGB image ready for recognition.
    """

    if image_array is None:
        raise ValueError("No image provided.")

    # Convert input to NumPy array
    image = np.asarray(image_array).astype(np.uint8)

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA image.")

    # Separate RGB and alpha channels
    rgb = image[:, :, :3]

    # Remove transparency by placing image on white background
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(rgb, 255, dtype=np.uint8)

        rgb = (
            rgb.astype(np.float32) * alpha
            + white.astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Detect handwriting pixels
    _, binary = cv2.threshold(
        gray,
        245,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Find handwriting boundaries
    points = cv2.findNonZero(binary)

    if points is None:
        raise ValueError("No handwriting detected.")

    x, y, w, h = cv2.boundingRect(points)

    # Crop handwriting
    cropped = rgb[y:y + h, x:x + w]

    # Add white padding
    cropped = cv2.copyMakeBorder(
        cropped,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    # Resize while preserving aspect ratio
    max_width = 1200
    max_height = 384

    height, width = cropped.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        cropped,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Return RGB image for AI models
    return Image.fromarray(resized).convert("RGB")
