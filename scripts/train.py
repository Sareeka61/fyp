import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from character_dataset import DevanagariOCRDataset
from ocr import label_to_index
from model import SimpleCNN

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = DevanagariOCRDataset("../data/character_ocr", label_to_index, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleCNN(num_classes=len(label_to_index))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "ocr_model.pth")
