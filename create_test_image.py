from PIL import Image, ImageDraw

image = Image.new("RGB", (100, 100), "white")
draw = ImageDraw.Draw(image)
draw.text((30, 20), "A", fill="black", stroke_width=2)

image.save("test_character.png")

print("Test image saved successfully!")
