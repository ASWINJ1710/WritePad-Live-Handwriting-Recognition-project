from PIL import Image, ImageDraw, ImageFont

from models.content_classifier import classify_content


# Create sample image
image = Image.new("RGB", (1000, 400), "white")
draw = ImageDraw.Draw(image)

try:
    font = ImageFont.truetype("arial.ttf", 65)
except OSError:
    font = ImageFont.load_default()

draw.text((50, 50), "Hello World 123", fill="black", font=font)

draw.text((50, 230), "2 + 3 = 5", fill="black", font=font)


# Crop text and math separately
text_image = image.crop((30, 30, 700, 150))
math_image = image.crop((30, 210, 700, 330))


# Test text classification
print("\nTesting TEXT:")
text_result = classify_content(text_image)
print(text_result)


# Test math classification
print("\nTesting MATH:")
math_result = classify_content(math_image)
print(math_result)