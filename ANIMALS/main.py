import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
import datetime
gpu="cuda"
device =torch.device(gpu)


train_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


train_data =datasets.ImageFolder(
    root= "animal_computer_vision/Dataset",
    transform=train_transformer
)

loader =DataLoader(
    train_data,
    64,
    True
)
num_classes = len(train_data.classes)
model =mobilenet_v2(weights ="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 20

for epoch in range(epochs):
    print(datetime.datetime.now())
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {running_loss/len(loader):.4f}")

