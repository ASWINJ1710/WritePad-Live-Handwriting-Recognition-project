
import numpy as np
import cv2

from PIL import Image

from utils.region_detection import detect_regions
from utils.character_segmentation import segment_characters

from models.text_recognition import recognize_text
from models.math_recognition import recognize_math
from models.character_recognition import predict_character
from models.symbol_recognition import predict_symbol


# Convert common HASYv2 LaTeX labels into readable symbols
SYMBOL_MAP = {
    r"\times": "×",
    r"\div": "÷",
    r"\pm": "±",
    r"\sqrt{}": "√",
    r"\%": "%",
    r"\neq": "≠",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\cdot": "·",
    r"\infty": "∞",
    r"\pi": "π",
    r"\alpha": "α",
    r"\beta": "β",
    r"\sum": "∑",
    r"\int": "∫",
}

MATH_OPERATORS = {
    "+", "-", "×", "÷", "=", "±",
    "≠", "≤", "≥", "%", "√"
}


def prepare_math_image(image):
    """Convert image to RGB and crop unnecessary white margins."""

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.fromarray(
            np.asarray(image).astype(np.uint8)
        ).convert("RGB")

    gray = np.array(image.convert("L"))
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


def is_equals_sign(image):
    """
    Detect an equals sign from two separate horizontal strokes.
    """

    gray = np.array(image.convert("L"))

    _, binary = cv2.threshold(
        gray, 245, 255, cv2.THRESH_BINARY_INV
    )

    row_counts = np.count_nonzero(binary, axis=1)

    active_rows = np.where(row_counts >= 2)[0]

    if len(active_rows) == 0:
        return False

    groups = []
    start = active_rows[0]
    previous = active_rows[0]

    for current in active_rows[1:]:
        if current > previous + 1:
            groups.append((start, previous))
            start = current

        previous = current

    groups.append((start, previous))

    if len(groups) != 2:
        return False

    widths = []

    for y1, y2 in groups:
        points = cv2.findNonZero(binary[y1:y2 + 1])

        if points is None:
            return False

        _, _, width, _ = cv2.boundingRect(points)
        widths.append(width)

    return (
        min(widths) >= 5
        and min(widths) / max(1, gray.shape[0]) >= 0.5
    )



def classify_crop(character_image):
    """
    Classify one handwritten character or mathematical symbol.
    """

    # 1. Detect equals sign geometrically.
    if is_equals_sign(character_image):
        return "="

    # 2. Get HASYv2 prediction.
    symbol_result = predict_symbol(character_image)

    symbol_label = symbol_result["symbol"]
    confidence = symbol_result["confidence"]

    readable_symbol = SYMBOL_MAP.get(
        symbol_label,
        symbol_label
    )

    # 3. Accept reliable digit predictions from HASYv2.
    if readable_symbol.isdigit() and confidence >= 40:
        return readable_symbol

    # 4. Plus sign: allow a lower threshold.
    if readable_symbol == "+" and confidence >= 30:
        return "+"

    # 5. Accept other mathematical operators.
    if (
        readable_symbol in MATH_OPERATORS
        and confidence >= 45
    ):
        return readable_symbol

    # 6. Fall back to LeNet.
    character_result = predict_character(character_image)

    if character_result is not None:
        return str(character_result["character"])

    return "?"


def recognize_combined_content(image):
    """
    Recognize a line containing letters, numbers and symbols.
    """

    regions = detect_regions(image)

    results = []

    for index, region in enumerate(regions, start=1):

        line_image = Image.fromarray(
            region["image"]
        ).convert("RGB")

        character_regions = segment_characters(line_image)

        predictions = []

        for character_region in character_regions:

            character_image = character_region["image"]

            try:
                prediction = classify_crop(character_image)
            except Exception:
                prediction = "?"

            predictions.append(prediction)

        # Keep operators visually separated
        formatted = []

        for prediction in predictions:

            if prediction in MATH_OPERATORS:
                if formatted and formatted[-1] != " ":
                    formatted.append(" ")

                formatted.append(prediction)
                formatted.append(" ")

            else:
                formatted.append(prediction)

        recognized_line = "".join(formatted).strip()

        results.append({
            "region": index,
            "image": line_image,
            "type": "text",
            "prediction": recognized_line,
            "x": region["x"],
            "y": region["y"],
            "width": region["width"],
            "height": region["height"]
        })

    return results


def recognize_mixed_content(image, mode="text"):

    results = []

    # MATH MODE: Existing Pix2Tex recognition
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

    # COMBINED MODE: LeNet5 + HASYv2
    if mode == "combined":
        return recognize_combined_content(image)

    # TEXT MODE: Existing TrOCR recognition
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
