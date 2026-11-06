
from PIL import Image, ImageDraw, ImageFont
from utils.preprocessing import preprocess_canvas
from utils.region_detection import detect_regions

print("Creating sample handwriting image...")

image = Image.new("RGB", (1000, 500), "white")
draw = ImageDraw.Draw(image)

try:
    font = ImageFont.truetype("arial.ttf", 65)
except OSError:
    font = ImageFont.load_default()

draw.text((50, 50), "Hello World 123", fill="black", font=font)
draw.text((50, 300), "2 + 3 = 5", fill="black", font=font)

print("Preprocessing image...")

processed_image = preprocess_canvas(image)

print("Detecting regions...")

regions = detect_regions(processed_image)

print("Total regions detected:", len(regions))

for index, region in enumerate(regions, start=1):
    print(f"Region {index}: {region['width']} x {region['height']}")

    Image.fromarray(region["image"]).save(
        f"outputs/region_{index}.png"
    )

print("Region detection test completed!")
