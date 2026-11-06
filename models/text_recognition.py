
import torch
import streamlit as st

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)


# Select GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_text_model():
    """
    Load the pretrained TrOCR handwritten text model.
    Model weights are downloaded automatically on first use.
    """

    model_name = "microsoft/trocr-base-handwritten"

    processor = TrOCRProcessor.from_pretrained(model_name)

    model = VisionEncoderDecoderModel.from_pretrained(
        model_name
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


def recognize_text(image):
    """
    Recognize handwritten text from a PIL image.
    """

    processor, model = load_text_model()

    # Convert image into model input
    pixel_values = processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to(DEVICE)

    # Generate handwritten text prediction
    with torch.inference_mode():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128
        )

    # Convert prediction into readable text
    recognized_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return recognized_text.strip()
