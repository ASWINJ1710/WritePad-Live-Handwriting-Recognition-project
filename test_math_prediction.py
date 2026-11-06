
from PIL import Image, ImageDraw, ImageFont
from models.math_recognition import recognize_math

# Create a sample mathematical equation image
image = Image.new("RGB", (800, 250), "white")
draw = ImageDraw.Draw(image)

try:
    font = ImageFont.truetype("arial.ttf", 90)
except OSError:
    font = ImageFont.load_default()

draw.text(
    (40, 60),
    "2 + 3 = 5",
    fill="black",
    font=font
)

print("Testing pix2tex...")

result = recognize_math(image)

print("LaTeX Result:", result)
