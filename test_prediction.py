
from PIL import Image, ImageDraw, ImageFont
from models.text_recognition import recognize_text

# Create a sample handwritten-style test image
image = Image.new("RGB", (1000, 250), "white")
draw = ImageDraw.Draw(image)

try:
    font = ImageFont.truetype("arial.ttf", 65)
except OSError:
    font = ImageFont.load_default()

draw.text(
    (30, 70),
    "Hello World 123",
    fill="black",
    font=font
)

print("Testing TrOCR...")

result = recognize_text(image)

print("Recognition Result:", result)
