import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
])


train_dataset = datasets.MNIST(
    root="mnist",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="mnist",
    train=False,
    transform=transform,
    download=True
)

print("Training samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

image,label = train_dataset[0]
print(label)
print(image.shape)

# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(train_dataset.classes[label])
# plt.axis("off")
# plt.show()


import torch.nn as nn
device = torch.device("cpu")
print(device)

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 5

for epoch in range(epochs):
    for images, labels in train_loader:

        optimizer.zero_grad()
        outputs = model(images)    # forward pass
        loss = criterion(outputs, labels)  # compute error
        loss.backward()            # backprop
        optimizer.step()           # update weights

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


image, label = test_dataset[0]

with torch.no_grad():
    output = model(image.unsqueeze(0))
    prediction = torch.argmax(output, dim=1).item()

print("True label:", label)
print("Predicted:", prediction)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Predicted: {prediction}")
plt.axis("off")
plt.show()
