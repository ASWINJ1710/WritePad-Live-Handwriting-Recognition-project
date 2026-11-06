import torch
from models.lenet5 import LeNet5

MODEL_PATH = "model_hub/fine_tuned/lenet_emnist_fresh.pth"
CLASS_PATH = "model_hub/fine_tuned/lenet_class_names.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading fresh LeNet-5 model...")

class_names = torch.load(
    CLASS_PATH,
    map_location="cpu",
    weights_only=False
)

model = LeNet5(num_classes=len(class_names)).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully!")
print("Device:", device)
print("Total classes:", len(class_names))

sample = torch.randn(1, 1, 28, 28).to(device)

with torch.no_grad():
    output = model(sample)
    predicted_index = output.argmax(dim=1).item()

print("Output shape:", output.shape)
print("Predicted index:", predicted_index)
print("Predicted character:", class_names[predicted_index])

print("\nFresh model test: SUCCESS")
