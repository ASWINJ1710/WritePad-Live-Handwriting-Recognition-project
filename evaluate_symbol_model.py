from pathlib import Path
import torch
from torch.utils.data import DataLoader
from symbol_dataset import HASYDataset
from models.symbol_cnn import SymbolCNN

ROOT = Path("datasets/hasyv2")
MODEL_PATH = Path("model_hub/fine_tuned/hasyv2_symbol_cnn.pth")
TEST_CSV = ROOT / "classification-task/fold-1/test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    checkpoint = torch.load(
        MODEL_PATH, map_location=device, weights_only=False
    )
except TypeError:
    checkpoint = torch.load(MODEL_PATH, map_location=device)

class_to_idx = checkpoint["class_to_idx"]

test_dataset = HASYDataset(
    ROOT,
    TEST_CSV,
    class_to_idx
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

model = SymbolCNN(num_classes=checkpoint["num_classes"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

correct = 0
total = 0
class_correct = [0] * len(class_to_idx)
class_total = [0] * len(class_to_idx)

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for label, prediction in zip(labels, predictions):
            idx = label.item()
            class_total[idx] += 1
            if label.item() == prediction.item():
                class_correct[idx] += 1

print("\n========== HASYv2 TEST RESULTS ==========")
print("Test images:", total)
print("Correct predictions:", correct)
print(f"Overall test accuracy: {100 * correct / total:.2f}%")

required = [
    "+", "-", r"\times", r"\div", "=",
    r"\pm", r"\sqrt{}", r"\%", "<", ">",
    "[", "]", "(", ")", r"\neq", r"\leq", r"\geq"
]

print("\n========== REQUIRED SYMBOL ACCURACY ==========")

for symbol in required:
    if symbol not in class_to_idx:
        print(f"{symbol}: NOT IN DATASET")
        continue

    idx = class_to_idx[symbol]
    count = class_total[idx]

    if count == 0:
        print(f"{symbol}: NO TEST SAMPLES")
    else:
        accuracy = 100 * class_correct[idx] / count
        print(f"{symbol}: {accuracy:.2f}% ({class_correct[idx]}/{count})")
