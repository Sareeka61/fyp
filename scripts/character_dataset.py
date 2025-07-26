import os
from PIL import Image
from torch.utils.data import Dataset

class DevanagariOCRDataset(Dataset):
    def __init__(self, root_dir, label_to_index, transform=None):
        self.samples = []
        self.transform = transform

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path) and label in label_to_index:
                for fname in os.listdir(label_path):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        path = os.path.join(label_path, fname)
                        self.samples.append((path, label_to_index[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label
