
import torch
import torch.nn.functional as F
import streamlit as st
import numpy as np

from PIL import Image
from torchvision import transforms

from models.symbol_cnn import SymbolCNN


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_symbol_model():

    checkpoint_path = "model_hub/fine_tuned/hasyv2_symbol_cnn.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    class_to_idx = checkpoint["class_to_idx"]

    idx_to_class = {
        index: label
        for label, index in class_to_idx.items()
    }

    model = SymbolCNN(
        num_classes=checkpoint["num_classes"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model, idx_to_class


def preprocess_symbol(image):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert("L")

    # Detect foreground pixels.
    image_array = np.array(image)

    foreground = image_array < 245

    if not foreground.any():
        return torch.zeros(1, 1, 32, 32)

    # Crop unnecessary white margins.
    ys, xs = np.where(foreground)

    image = image.crop((
        max(0, xs.min()),
        max(0, ys.min()),
        min(image.width, xs.max() + 1),
        min(image.height, ys.max() + 1)
    ))

    # Preserve aspect ratio.
    image.thumbnail((24, 24), Image.Resampling.LANCZOS)

    # Create white 32x32 canvas.
    canvas = Image.new("L", (32, 32), 255)

    # Centre the symbol.
    x = (32 - image.width) // 2
    y = (32 - image.height) // 2

    canvas.paste(image, (x, y))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(canvas)

    return tensor.unsqueeze(0)


def predict_symbol(image):

    model, idx_to_class = load_symbol_model()

    tensor = preprocess_symbol(image).to(DEVICE)

    with torch.inference_mode():

        output = model(tensor)

        probabilities = F.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    index = prediction.item()

    return {
        "symbol": idx_to_class[index],
        "confidence": round(confidence.item() * 100, 2)
    }
