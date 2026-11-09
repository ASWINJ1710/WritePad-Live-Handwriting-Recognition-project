import torch
import torch.nn.functional as F
import streamlit as st
import numpy as np
import cv2

from PIL import Image

from models.lenet5 import LeNet5


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_character_model():

    model = LeNet5(num_classes=62)

    checkpoint_path = "model_hub/fine_tuned/lenet_emnist_fresh.pth"

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    class_names = torch.load(
        "model_hub/fine_tuned/lenet_class_names.pth",
        map_location="cpu",
        weights_only=False
    )

    return model, class_names


def preprocess_character(image):

    image = image.convert("L")
    image_array = np.array(image)

    # Convert black handwriting into white foreground
    _, binary = cv2.threshold(
        image_array,
        245,
        255,
        cv2.THRESH_BINARY_INV
    )

    points = cv2.findNonZero(binary)

    if points is None:
        return None

    # Crop character
    x, y, w, h = cv2.boundingRect(points)
    character = binary[y:y + h, x:x + w]

    # Match approximate character dimensions in training images
    scale = min(11 / w, 14 / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    character = cv2.resize(
        character,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Create 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Center character
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = character

    # White background, black character
    canvas = 255 - canvas

    # Convert to normalized tensor
    tensor = torch.from_numpy(canvas).float() / 255.0
    tensor = (tensor - 0.5) / 0.5

    return tensor.unsqueeze(0).unsqueeze(0)


def predict_character(image):

    model, class_names = load_character_model()

    tensor = preprocess_character(image)

    if tensor is None:
        return None

    tensor = tensor.to(DEVICE)

    with torch.inference_mode():

        output = model(tensor)
        probabilities = F.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    index = prediction.item()

    return {
        "character": class_names[index],
        "confidence": round(confidence.item() * 100, 2)
}