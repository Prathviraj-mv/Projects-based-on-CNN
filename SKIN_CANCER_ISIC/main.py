import torch
from torchvision import datasets,transforms
from  torchvision.models import densenet121
from torch.utils.data import DataLoader
import datetime as dt

test_file ="Test"
train_file ="Train"

Laptop ="cuda"
device =torch.device(Laptop)

train_tf =transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ]
)
test_tf =transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ]
)

train_data =datasets.ImageFolder(root=train_file,transform=train_tf)
test_data =datasets.ImageFolder(root=test_file,transform=test_tf)

model_data = DataLoader(train_data,batch_size=32,shuffle =True)
model_test = DataLoader(test_data,batch_size=32,shuffle=False)

model =densenet121(pretrained =True)

num_classes = len(train_data.classes)

model.classifier = torch.nn.Linear(
    model.classifier.in_features,
    num_classes
)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr =0.001)
crit =torch.nn.CrossEntropyLoss()

epoch =10

for epoch in range(epoch):
    print(dt.datetime.now())
    correct =0
    r_loss =0
    for image,label in model_data:
        image =image.to(device)
        label =label.to(device)

        optimizer.zero_grad()
        op =model(image)
        loss = crit(op,label)
        loss.backward()
        optimizer.step()
        r_loss +=loss.item()
        _, preds = torch.max(op, 1)
        correct += (preds == label).sum().item()

    acc = correct / len(train_data)

    print(f"Epoch {epoch + 1}: Loss={r_loss:.4f}, Acc={acc:.4f}")

model.eval()

correct = 0

with torch.no_grad():
    for images, labels in model_test:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()

test_acc = correct / len(test_data)
print("Test Accuracy:", test_acc)
