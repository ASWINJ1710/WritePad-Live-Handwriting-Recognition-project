
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "microsoft/trocr-base-handwritten"

print("Loading TrOCR processor...")

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

print("Loading TrOCR model...")

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

print("TrOCR loaded successfully!")
print("Device:", device)

