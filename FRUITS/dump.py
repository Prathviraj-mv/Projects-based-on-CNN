import torch
import torch.nn as nn
from torchvision.models import resnet18,ResNet18_Weights
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import joblib
from datetime import datetime
transform =transforms.Compose([transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize(
                               mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]
                               )])

train = datasets.ImageFolder(root ="fruits/train",transform=transform)
test = datasets.ImageFolder(root ="fruits/test_",transform=transform)
train_l = DataLoader(train,batch_size=64,shuffle=True)
test_l = DataLoader(test,batch_size=64,shuffle=False)
device = torch.device("cuda")
image,label = train[0]
print(f"device = {device} \nimage ={image.shape} label= {label}\nclasses ={train.classes}")

weights =ResNet18_Weights.DEFAULT
model =resnet18(weights=weights)
num_classes = len(train.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


epochs = 5

for epoch in range(epochs):
    print("Time:", datetime.now().strftime("%H:%M:%S"))
    model.train()
    correct = 0
    total =0


    for images,labels in train_l:
        images =images.to(device)
        labels =labels.to(device)
        optimizer.zero_grad()
        outputs =model(images)
        loss =criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        waste, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        train_acc = 100 * correct / total
    print(f" epoch = {epoch + 1} loss = {loss.item()}, accuracy = {train_acc}")




joblib.dump({"model_state": model.state_dict(), "class_names": train.classes}, "fruits.pkl")

model.eval()

image, label = test[0]
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    pred = output.argmax(1).item()

print("True label:", train.classes[label])
print("Predicted :", train.classes[pred])
