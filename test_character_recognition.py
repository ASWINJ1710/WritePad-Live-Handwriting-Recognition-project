from PIL import Image, ImageDraw
from models.character_recognition import predict_character

image = Image.new("RGB", (100, 100), "white")
draw = ImageDraw.Draw(image)

draw.text((30, 20), "A", fill="black", stroke_width=2)

result = predict_character(image)

print("Prediction:", result)
