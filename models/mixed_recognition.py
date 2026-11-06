from PIL import Image
from utils.region_detection import detect_regions
from models.text_recognition import recognize_text

def recognize_mixed_content(image):
    regions = detect_regions(image)
    results = []

    for index, region in enumerate(regions, start=1):
        region_image = Image.fromarray(region["image"]).convert("RGB")

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

    return results
