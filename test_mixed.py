from PIL import Image, ImageDraw, ImageFont

from utils.preprocessing import preprocess_canvas
from models.mixed_recognition import recognize_mixed_content


# Create sample image
image = Image.new("RGB", (1000, 500), "white")
draw = ImageDraw.Draw(image)

try:
    font = ImageFont.truetype("arial.ttf", 65)
except OSError:
    font = ImageFont.load_default()

# Write text and equation
draw.text((50, 50), "Hello World 123", fill="black", font=font)
draw.text((50, 300), "2 + 3 = 5", fill="black", font=font)


# Preprocess image
processed_image = preprocess_canvas(image)


# Automatic mixed recognition
results = recognize_mixed_content(processed_image)


print("\nTotal regions:", len(results))

for result in results:

    print("\n--------------------------")
    print("Region:", result["region"])
    print("Detected Type:", result["type"])
    print("Prediction:", result["prediction"])
    print("--------------------------")


print("\nAutomatic mixed recognition completed!")