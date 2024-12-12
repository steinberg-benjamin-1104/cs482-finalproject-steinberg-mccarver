import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getInputPaths():
    root = tk.Tk()
    root.withdraw()
    print("Please select the folder containing the images.")
    imagesPath = filedialog.askdirectory(title="Select Images Folder")
    print("Please select the folder containing the labels.")
    labelsPath = filedialog.askdirectory(title="Select Labels Folder")
    return imagesPath, labelsPath

def get_downloads_folder():
    return os.path.join(os.path.expanduser("~"), "Downloads")

class Door:
    def __init__(self, box, category="GenericDoor"):
        self.box = box
        self.category = category

    def __repr__(self):
        return f"{self.category}(box={self.box})"

class RegularDoor(Door):
    def __init__(self, box):
        super().__init__(box, category="RegularDoor")

class CabinetDoor(Door):
    def __init__(self, box):
        super().__init__(box, category="CabinetDoor")

class RefrigeratorDoor(Door):
    def __init__(self, box):
        super().__init__(box, category="RefrigeratorDoor")

class Handle:
    def __init__(self, box):
        self.box = box

    def __repr__(self):
        return f"Handle(box={self.box})"

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x

class MyCNNClassifier:
    def __init__(self, imagesPath, labelsPath):
        self.imagesPath = imagesPath
        self.labelsPath = labelsPath
        self.x = None
        self.y = None
        self.train_loader = None
        self.test_loader = None

    def preprocess_images(self):
        image_files = [f for f in os.listdir(self.imagesPath) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"Total number of image files: {len(image_files)}")

        features = []
        targets = []

        for ifile in image_files:
            print(f"Processing image file: {ifile}")
            try:
                image = cv2.imread(os.path.join(self.imagesPath, ifile), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Skipping file due to loading error: {ifile}")
                    continue

                imageName, _ = os.path.splitext(ifile)
                lfile = os.path.join(self.labelsPath, f"{imageName}.txt")

                objects = []

                if os.path.exists(lfile) and os.path.getsize(lfile) > 0:
                    with open(lfile, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                obj_class, x, y, w, h = map(float, parts)
                                obj_class = int(obj_class)

                                img_height, img_width = image.shape
                                x_min = int((x - w / 2) * img_width)
                                y_min = int((y - h / 2) * img_height)
                                x_max = int((x + w / 2) * img_width)
                                y_max = int((y + h / 2) * img_height)

                                x_min = max(0, x_min)
                                y_min = max(0, y_min)
                                x_max = min(img_width, x_max)
                                y_max = min(img_height, y_max)

                                box = (x_min, y_min, x_max, y_max)

                                if obj_class == 0:
                                    obj = RegularDoor(box)
                                elif obj_class == 1:
                                    obj = Handle(box)
                                elif obj_class == 2:
                                    obj = CabinetDoor(box)
                                elif obj_class == 3:
                                    obj = RefrigeratorDoor(box)
                                else:
                                    print(f"Unknown object class {obj_class}, skipping.")
                                    continue

                                objects.append(obj)
                                print(f"Parsed object: {obj}")

                    if objects:
                        for obj in objects:
                            x_min, y_min, x_max, y_max = obj.box
                            cropped_image = image[y_min:y_max, x_min:x_max]

                            if cropped_image.size == 0:
                                print(f"Invalid cropped region for object: {obj}, skipping.")
                                continue

                            cropped_image = cv2.resize(cropped_image, (64, 64))
                            cropped_image = cropped_image / 255.0

                            features.append(cropped_image)
                            targets.append(obj_class)
            except Exception as e:
                print(f"Error processing file {ifile}: {e}")
                continue

        self.x = np.array(features, dtype=np.float32).reshape(-1, 1, 64, 64)
        self.y = np.array(targets, dtype=np.int64)

        print(f"Total features extracted: {len(self.x)}")
        print(f"Total targets extracted: {len(self.y)}")

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train_cnn(model, train_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

def evaluate_cnn(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["RegularDoor", "Handle", "CabinetDoor", "RefrigeratorDoor"])
    print(f"Accuracy: {acc:.4f}")
    print(report)

if __name__ == "__main__":
    imagesPath, labelsPath = getInputPaths()

    classifier = MyCNNClassifier(imagesPath, labelsPath)
    classifier.preprocess_images()

    cnn_model = CNN().to(device)
    train_cnn(cnn_model, classifier.train_loader, num_epochs=10)
    evaluate_cnn(cnn_model, classifier.test_loader)
