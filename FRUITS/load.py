import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import joblib
from datetime import datetime

device = torch.device("cuda")

ckpt = joblib.load("fruits.pkl")
classes = ckpt["class_names"]

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(ckpt["model_state"])

model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

print(f"time ={datetime.now().strftime("%H:%M:%S")}")
img = Image.open("0014.jpg").convert("RGB")
img = transform(img).unsqueeze(0).to(device)


with torch.no_grad():
    out = model(img)
    pred = out.argmax(1)

print(classes[pred.item()])
