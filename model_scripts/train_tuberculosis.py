# TB Chest X-ray Classification (Normal vs Tuberculosis)
# End-to-end training + evaluation + inference

import os
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_curve
import cv2

def crop_lung_region(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # normalize contrast (important)
    img = cv2.equalizeHist(img)

    # threshold to isolate chest area
    blur = cv2.GaussianBlur(img,(5,5),0)
    _,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # largest contour = chest region
    contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h, x:x+w]

    crop = cv2.resize(crop,(224,224))
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(crop)

# =========================
# 1. CONFIG
# =========================
DATASET_DIR = "/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "/kaggle/working/tb1_model.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# 2. LOAD FILE PATHS
# =========================
def load_image_paths(base_dir):
    normal_dir = os.path.join(base_dir, "Normal")
    tb_dir = os.path.join(base_dir, "Tuberculosis")

    normal = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.lower().endswith(('png','jpg','jpeg'))]
    tb = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.lower().endswith(('png','jpg','jpeg'))]

    X = normal + tb
    y = [0]*len(normal) + [1]*len(tb)

    return X, y

X, y = load_image_paths(DATASET_DIR)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# =========================
# 3. DATASET CLASS
# =========================
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=5, scale=(0.9,1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

])

class TBXrayDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = crop_lung_region(self.paths[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

train_ds = TBXrayDataset(X_train, y_train, train_tfms)
val_ds = TBXrayDataset(X_val, y_val, val_tfms)
test_ds = TBXrayDataset(X_test, y_test, val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# =========================
# 4. MODEL (TRANSFER LEARNING)
# =========================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# 5. TRAIN FUNCTION
# =========================
def train_epoch(loader):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss/len(loader)

# =========================
# 6. VALIDATION
# =========================
def eval_model(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = torch.sigmoid(model(imgs)).cpu().numpy()
            preds.extend(outputs>0.5)
            trues.extend(labels.numpy())
    return accuracy_score(trues, preds)

# =========================
# 7. TRAIN LOOP
# =========================
best_acc = 0
for epoch in range(EPOCHS):
    loss = train_epoch(train_loader)
    val_acc = eval_model(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("Model Saved!")

# =========================
# 8. TEST EVALUATION
# =========================
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

preds, trues = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = torch.sigmoid(model(imgs)).cpu().numpy()
        preds.extend(outputs>0.5)
        trues.extend(labels.numpy())

print("\nTest Accuracy:", accuracy_score(trues, preds))
print("\nClassification Report:\n", classification_report(trues, preds))
print("\nConfusion Matrix:\n", confusion_matrix(trues, preds))


probs = []
trues = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = torch.sigmoid(model(imgs)).cpu().numpy()
        probs.extend(outputs)
        trues.extend(labels.numpy())

fpr, tpr, thresholds = roc_curve(trues, probs)

# choose high sensitivity threshold
best_idx = np.argmax(tpr - fpr)
best_threshold = thresholds[best_idx]

print("Recommended Medical Threshold:", best_threshold)


# =========================
# 9. SINGLE IMAGE PREDICTION
# =========================
def predict_image(img_path, threshold):

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    img = crop_lung_region(img_path)
    img = val_tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(img)).item()

    label = "Tuberculosis" if prob > threshold else "Normal"
    confidence = interpret_probability(prob)

    print(f"\nPrediction: {label}")
    print(f"TB Probability: {prob*100:.2f}%")
    print(f"Confidence Level: {confidence}")


def interpret_probability(p):

    if p < 0.20:
        return "Very Low Risk"
    elif p < 0.40:
        return "Low Risk"
    elif p < 0.60:
        return "Borderline"
    elif p < 0.80:
        return "High Risk"
    else:
        return "Very High Risk"
