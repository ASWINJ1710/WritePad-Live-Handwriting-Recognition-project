
import cv2
import numpy as np
from PIL import Image


def segment_characters(image, min_area=8, padding=5):
    """
    Segment handwritten characters and mathematical symbols.

    Merges vertically separated components when they likely
    belong to the same character, including equals signs.
    Returns crops in left-to-right order.
    """

    # Convert input to RGB
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        array = np.asarray(image)

        if array.ndim == 2:
            rgb = np.array(
                Image.fromarray(array).convert("RGB")
            )
        elif array.shape[2] == 4:
            rgb = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
        else:
            rgb = array.copy()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Convert handwriting into foreground
    _, binary = cv2.threshold(
        gray, 245, 255, cv2.THRESH_BINARY_INV
    )

    # Detect connected components
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
    )

    components = []

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue

        components.append({
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + w),
            "y2": int(y + h),
            "area": int(area)
        })

    if not components:
        return []

    # Merge disconnected strokes belonging to one symbol
    changed = True

    while changed:
        changed = False
        merged = []
        used = [False] * len(components)

        for i, first in enumerate(components):

            if used[i]:
                continue

            current = first.copy()
            used[i] = True

            for j, second in enumerate(components):

                if used[j] or i == j:
                    continue

                # Horizontal overlap
                overlap = max(
                    0,
                    min(current["x2"], second["x2"])
                    - max(current["x1"], second["x1"])
                )

                current_width = (
                    current["x2"] - current["x1"]
                )

                second_width = (
                    second["x2"] - second["x1"]
                )

                min_width = min(
                    current_width,
                    second_width
                )

                # Vertical gap
                vertical_gap = max(
                    0,
                    max(current["y1"], second["y1"])
                    - min(current["y2"], second["y2"])
                )

                current_height = (
                    current["y2"] - current["y1"]
                )

                second_height = (
                    second["y2"] - second["y1"]
                )

                max_height = max(
                    current_height,
                    second_height
                )

                # Normal grouping condition
                enough_horizontal_overlap = (
                    min_width > 0
                    and overlap / min_width >= 0.5
                )

                close_vertically = (
                    vertical_gap <= max(
                        18,
                        0.8 * max_height
                    )
                )

                # Special condition for equals sign (=)
                # Both components must be long horizontal strokes.
                both_horizontal_strokes = (
                    current_width >= 3 * max(1, current_height)
                    and second_width >= 3 * max(1, second_height)
                )

                equals_sign_condition = (
                    both_horizontal_strokes
                    and min_width > 0
                    and overlap / min_width >= 0.65
                    and vertical_gap <= 45
                )

                should_merge = (
                    enough_horizontal_overlap
                    and (
                        close_vertically
                        or equals_sign_condition
                    )
                )

                if should_merge:

                    current["x1"] = min(
                        current["x1"],
                        second["x1"]
                    )

                    current["y1"] = min(
                        current["y1"],
                        second["y1"]
                    )

                    current["x2"] = max(
                        current["x2"],
                        second["x2"]
                    )

                    current["y2"] = max(
                        current["y2"],
                        second["y2"]
                    )

                    current["area"] += second["area"]

                    used[j] = True
                    changed = True

            merged.append(current)

        components = merged

    # Sort characters from left to right
    components.sort(key=lambda item: item["x1"])

    character_crops = []

    image_height, image_width = gray.shape

    for component in components:

        x1 = max(0, component["x1"] - padding)
        y1 = max(0, component["y1"] - padding)

        x2 = min(
            image_width,
            component["x2"] + padding
        )

        y2 = min(
            image_height,
            component["y2"] + padding
        )

        crop = rgb[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        character_crops.append({
            "image": Image.fromarray(crop).convert("RGB"),
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1
        })

    return character_crops
