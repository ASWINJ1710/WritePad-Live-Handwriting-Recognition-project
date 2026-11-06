import torch
from models.lenet5 import LeNet5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeNet5(num_classes=62)

checkpoint = torch.load(
    "model_hub/pre_trained/lenet.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint, strict=True)

model = model.to(device)
model.eval()

print("Pretrained LeNet loaded successfully!")
print("Device:", device)

sample = torch.randn(1, 1, 28, 28).to(device)

with torch.no_grad():
    output = model(sample)

print("Prediction output:", output.shape)
print("Predicted class index:", output.argmax(dim=1).item())
