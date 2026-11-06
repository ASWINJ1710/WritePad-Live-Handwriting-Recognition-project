import torch
from models.lenet5 import LeNet5

checkpoint_path = "model_hub/pre_trained/lenet.pth"

print("Loading pretrained checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

# Handle common checkpoint formats
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# Handle DataParallel checkpoints
state_dict = {
    key.replace("module.", "", 1): value
    for key, value in state_dict.items()
}

model = LeNet5(num_classes=62)

try:
    model.load_state_dict(state_dict, strict=True)
    print("\nSUCCESS: All pretrained weights loaded!")
except RuntimeError as error:
    print("\nCheckpoint architecture mismatch:")
    print(error)
    raise SystemExit(1)

model.eval()

# Test one dummy handwriting image
dummy_input = torch.randn(1, 1, 28, 28)

with torch.no_grad():
    output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)
print("Number of classes:", output.shape[1])
