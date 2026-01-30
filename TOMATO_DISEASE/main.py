import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import datetime
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda")


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_location = "tomato/train"
test_location = "tomato/val"

train = datasets.ImageFolder(root=train_location, transform=train_transform)
test = datasets.ImageFolder(root=test_location, transform=val_transform)

#
train_l = DataLoader(train, batch_size=32, shuffle=True)
test_l = DataLoader(test, batch_size=32, shuffle=False)

image, label = train[0]
print(f"Label ={label} , size = {image.shape}, classes ={train.classes}")
model = models.densenet121(pretrained=True)

for param in model.features.parameters():
    param.requires_grad = False

class_len = len(train.classes)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, class_len)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)


epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    print(datetime.datetime.now())

    for images, labels in train_l:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_l)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_l:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")
