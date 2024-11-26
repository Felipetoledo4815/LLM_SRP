import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from vlm.SpaceLlavaWrapper import SpaceLlavaWrapper
import xml.etree.ElementTree as ET

# Step 1: Initialize the SpaceLlava model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "spacellava"
vlm = SpaceLlavaWrapper(
    clip_path="./models/spacellava/mmproj-model-f16.gguf",
    model_path="./models/spacellava/ggml-model-q4_0.gguf"
)

vlm = vlm.spacellava

def get_object_names(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    object_names = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        object_names.append(name)

    return object_names

def get_image_list(directory, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


class ImageDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]["image_file_path"]
        label = self.data[idx]["labels"]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label


# Replace with your actual data
image_directory = "/home/radowanredoy/Desktop/rotation1/LLM_SRP/vlm/data/archive/images"
annotation_directory = "/home/radowanredoy/Desktop/rotation1/LLM_SRP/vlm/data/archive/annotations/"
image_paths = get_image_list(image_directory)
data = []
labels = []
for image in image_paths:
    label_data = get_object_names(annotation_directory + image.split('/')[-1].split('.')[0] + '.xml')
    data.append({
        'labels': label_data,
        'image_file_path': image
    })
    labels.extend(label_data)

# Preprocess images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust to SpaceLlava's expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageDataset(data, transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Map labels to indices
unique_labels = list(set(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
num_classes = len(unique_labels)

print(num_classes)

# Step 3: Define the Training Loop
optimizer = torch.optim.AdamW(vlm.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training configuration
num_epochs = 5
#
# for epoch in range(num_epochs):
#     vlm.train()
#     total_loss = 0
#
#     for images, labels in data_loader:
#         images = images.to(device)
#         label_indices = torch.tensor([label_to_index[label] for label in labels]).to(device)
#
#         # Forward pass through the SpaceLlava model
#         logits = vlm(images)
#
#         # Compute loss
#         loss = criterion(logits, label_indices)
#         total_loss += loss.item()
#
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")
#
# # Save the fine-tuned model
# torch.save(vlm.state_dict(), "fine_tuned_spacellava.pth")
# print("Training complete. Model saved!")
