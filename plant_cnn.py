import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Пристрій:", device)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print("Розмір навчальної вибірки:", len(train_data))
print("Розмір тестової вибірки:", len(test_data))

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 16 * 16, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Епоха {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), losses, marker="o", color="blue")
plt.xlabel("Епоха")
plt.ylabel("Loss")
plt.title("Залежність втрат від епох")
plt.xticks(range(1, epochs + 1))
plt.grid(True)
plt.show()

torch.save(model.state_dict(), "plant_cnn.pth")
print("Модель збережено у файл plant_cnn.pth")

model.eval()
correct = 0
total = 0
test_loss = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
avg_test_loss = test_loss / len(test_loader)

print()
print(f"{'Метрика':<10}{'Значення'}")
print(f"{'Accuracy':<10}{accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"{'Loss':<10}{avg_test_loss:.4f}")
