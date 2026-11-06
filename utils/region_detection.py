
import cv2
import numpy as np


def detect_regions(image, min_area=10):
    """
    Detect complete horizontal writing lines.
    """

    image_array = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # Convert handwriting into white foreground
    _, binary = cv2.threshold(
        gray,
        245,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Connect vertically separated parts of characters
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 15)
    )

    connected = cv2.dilate(
        binary,
        kernel,
        iterations=1
    )

    # Find active horizontal rows
    active_rows = np.any(connected > 0, axis=1)

    # Find continuous row groups
    row_indices = np.where(active_rows)[0]

    if len(row_indices) == 0:
        return []

    groups = []
    start = row_indices[0]
    previous = row_indices[0]

    for current in row_indices[1:]:

        if current > previous + 1:
            groups.append((start, previous))
            start = current

        previous = current

    groups.append((start, previous))

    regions = []

    for y1, y2 in groups:

        # Find horizontal boundaries
        line = binary[y1:y2 + 1, :]

        points = cv2.findNonZero(line)

        if points is None:
            continue

        x, _, w, _ = cv2.boundingRect(points)

        if w * (y2 - y1 + 1) < min_area:
            continue

        # Add padding
        padding = 10

        x1 = max(0, x - padding)
        x2 = min(image_array.shape[1], x + w + padding)

        top = max(0, y1 - padding)
        bottom = min(image_array.shape[0], y2 + 1 + padding)

        crop = image_array[top:bottom, x1:x2]

        regions.append({
            "image": crop,
            "x": x1,
            "y": top,
            "width": crop.shape[1],
            "height": crop.shape[0]
        })

    return regions
