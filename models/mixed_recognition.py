import numpy as np
from PIL import Image

from utils.region_detection import detect_regions
from models.text_recognition import recognize_text
from models.math_recognition import recognize_math


def prepare_math_image(image):
    """Convert image to RGB and crop unnecessary white margins."""

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.fromarray(
            np.asarray(image).astype(np.uint8)
        ).convert("RGB")

    gray = np.array(image.convert("L"))

    # Find all non-white handwriting pixels
    mask = gray < 245

    if not np.any(mask):
        return image

    ys, xs = np.where(mask)

    padding = 25

    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(image.width, int(xs.max()) + padding + 1)
    bottom = min(image.height, int(ys.max()) + padding + 1)

    return image.crop((left, top, right, bottom))


def recognize_mixed_content(image, mode="text"):

    results = []

    # MATH MODE: Process the complete equation together
    if mode == "math":

        try:
            math_image = prepare_math_image(image)

            prediction = recognize_math(math_image)

            results.append({
                "region": 1,
                "image": math_image,
                "type": "math",
                "prediction": prediction,
                "x": 0,
                "y": 0,
                "width": math_image.width,
                "height": math_image.height
            })

        except Exception as e:
            raise RuntimeError(
                f"Math recognition failed: {e}"
            ) from e

        return results

    # TEXT MODE: Keep existing region-based recognition
    regions = detect_regions(image)

    for index, region in enumerate(regions, start=1):

        region_image = Image.fromarray(
            region["image"]
        ).convert("RGB")

        try:
            prediction = recognize_text(region_image)

            results.append({
                "region": index,
                "image": region_image,
                "type": "text",
                "prediction": prediction,
                "x": region["x"],
                "y": region["y"],
                "width": region["width"],
                "height": region["height"]
            })

        except Exception as e:
            results.append({
                "region": index,
                "image": region_image,
                "type": "text",
                "prediction": f"Recognition error: {e}",
                "x": region["x"],
                "y": region["y"],
                "width": region["width"],
                "height": region["height"]
            })

    return results
