import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# FashionMNIST (datasets + DataLoaders) lives in data.py
from data import train_data, test_data, train_loader, test_loader
from data import FASHION_CLASS_NAMES as class_names

# Reproducibility: same seed -> same shuffles, same initial weights.
torch.manual_seed(1337)

# CNNs love a GPU. 'cuda' = an NVIDIA GPU; 'cpu' = your processor.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("torch version:", torch.__version__)
print("Using device :", device)

## ---------------------------- Convolutional neural net  ------------------------------------------------


class CNN(nn.Module):
    """
    Small convolutional network for 28x28 grayscale images (e.g. MNIST).

    Architecture:
        conv(1 -> 16, 3x3) + ReLU + max-pool  -> (B, 16, 14, 14)
        conv(16 -> 32, 3x3) + ReLU + max-pool -> (B, 32, 7, 7)
        flatten -> fc(1568 -> 128) + ReLU -> fc(128 -> 10)

    Input:  (B, 1, 28, 28) tensor.
    Output: (B, 10) raw logits, one per class; use CrossEntropyLoss for
    training or torch.softmax to get class probabilities.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # -> (B, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))     # -> (B, 32,  7,  7)
        x = torch.flatten(x, 1)                  # -> (B, 1568)
        x = F.relu(self.fc1(x))                  # -> (B, 128)
        x = self.fc2(x)                          # -> (B, 10) logits
        return x


model = CNN().to(device)
print(model)
print()
## ---------------------------- accuracy helper ------------------------------------------------


@torch.no_grad()
def accuracy(model, loader):
    model.eval()                          # evaluation mode
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        preds = logits.argmax(dim=1)      # the class with the highest score
        correct += (preds == y).sum().item()
        total   += y.size(0)
    model.train()                         # back to training mode
    return correct / total

## ---------------------------- Training loop ------------------------------------------------
def train(model, train_loader, epochs=3, lr=1e-3):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()                # 1. clear gradients
            logits = model(X)                    # 2. forward
            loss = loss_fn(logits, y)            # 3. loss
            loss.backward()                      # 4. backward
            optimizer.step()                     # 5. update
            loss_history.append(loss.item())
        print(f"epoch {epoch+1}/{epochs} | last loss {loss.item():.4f} "
              f"| train acc {accuracy(model, train_loader):.3f} "
              f"| test acc {accuracy(model, test_loader):.3f}")
    return loss_history
## ---------------------------- Prediction batch ------------------------------------------------
model.eval()
X, y = next(iter(test_loader))
with torch.no_grad():
    preds = model(X.to(device)).argmax(dim=1).cpu()
print(preds[:12])
print([class_names[p] for p in preds[:6]])

## ---------------------------- Feature maps ------------------------------------------------


img, label = test_data[0]
with torch.no_grad():
    fmaps = F.relu(model.conv1(img.unsqueeze(0).to(device)))[0].cpu()   # (16, 28, 28)

fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
for ax, fm in zip(axes.flat, fmaps):
    ax.imshow(fm, cmap="gray")
    ax.axis("off")
plt.suptitle(f"conv1 feature maps for a '{class_names[label]}'", y=1.04)
plt.tight_layout()
plt.show()