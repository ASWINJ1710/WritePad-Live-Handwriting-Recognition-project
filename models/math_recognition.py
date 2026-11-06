
import streamlit as st
import torch

from pix2tex.cli import LatexOCR


@st.cache_resource
def load_math_model():
    """
    Load the pretrained pix2tex mathematical OCR model.
    """

    model = LatexOCR()
    return model


def recognize_math(image):
    """
    Convert a mathematical equation image into LaTeX.
    """

    model = load_math_model()

    # Ensure the input is a PIL image
    image = image.convert("RGB")

    # Predict LaTeX equation
    with torch.inference_mode():
        latex_output = model(image)

    return latex_output.strip()
