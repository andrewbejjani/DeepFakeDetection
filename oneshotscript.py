import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import kaggle
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
DATASET_NAME = "ayushmandatta1/deepdetect-2025" 
DOWNLOAD_ROOT = "./deepdetect_data"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- 2. DOWNLOAD DATASET ---
if not os.path.exists(DOWNLOAD_ROOT):
    print(f"Downloading {DATASET_NAME}...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET_NAME, path=DOWNLOAD_ROOT, unzip=True)
    print("Download complete.")
else:
    print("Dataset already downloaded.")

# --- 3. DATA PREPARATION (FIXED) ---
# Helper function to find the actual data folders if they are nested
def find_folder(root_dir, target_name):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_name in dirnames:
            return os.path.join(dirpath, target_name)
    raise FileNotFoundError(f"Could not find folder '{target_name}' inside {root_dir}")

# Locate the 'train' and 'test' folders automatically
try:
    TRAIN_PATH = find_folder(DOWNLOAD_ROOT, 'train')
    VAL_PATH = find_folder(DOWNLOAD_ROOT, 'test')
    print(f"Found Train Path: {TRAIN_PATH}")
    print(f"Found Test Path:  {VAL_PATH}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"Current structure: {os.listdir(DOWNLOAD_ROOT)}")
    exit()

# Define Transforms (Resize to 224x224 for EfficientNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets specifically from their folders
train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
val_dataset = datasets.ImageFolder(root=VAL_PATH, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes: {train_dataset.classes}") # Should be ['Fake', 'Real']

# --- 4. BUILD MODEL (Transfer Learning) ---


# Load Pre-trained EfficientNet-B0
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Freeze feature layers (We only want to train the classifier)
for param in model.features.parameters():
    param.requires_grad = False

# Replace the classifier head
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2) # Output: 2 classes (Real/Fake)

model = model.to(DEVICE)

# --- 5. TRAINING LOOP WITH VALIDATION ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")

for epoch in range(EPOCHS):
    # A. Training Phase
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total

    # B. Validation Phase (The "Test" Check)
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Acc: {val_acc:.2f}%")

# --- 6. SAVE MODEL ---
torch.save(model.state_dict(), "deepdetect_2025_model.pth")
print("Model saved successfully.")

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
WARMUP_EPOCHS = 5      # Phase 1: Train classifier only
FINETUNE_EPOCHS = 10   # Phase 2: Train backbone + classifier
TOTAL_EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS

# --- 2. STRONG AUGMENTATION (REGULARIZATION) ---
# Deepfake detection requires robustness to quality loss and slight geometry changes.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    
    # Geometric Augmentations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Simulate slight camera shifts
    
    # Quality/Color Augmentations (Crucial for Deepfakes)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Simulate blur/compression artifacts
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. DATA LOADERS ---
train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=train_transform)
val_dataset = datasets.ImageFolder(root=VAL_PATH, transform=val_transform)

# num_workers=0 avoids Windows/VSCode freeze issues
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Classes: {train_dataset.classes}")

# --- 4. MODEL SETUP ---
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# A. INITIAL FREEZE: Freeze EVERYTHING first
for param in model.parameters():
    param.requires_grad = False

# B. MODIFY CLASSIFIER: High Dropout as requested (0.4 - 0.5)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5), # Stronger dropout
    nn.Linear(num_features, 2)
)

model = model.to(DEVICE)

# --- 5. LOSS FUNCTION (With Label Smoothing) ---
# Label smoothing prevents the model from being "too confident" (overfitting)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# --- HELPER FUNCTION FOR TRAINING ---
def train_one_epoch(model, loader, optimizer, phase_name):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    acc = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, acc

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels) # Calculate val loss too
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    avg_loss = val_loss / len(loader)
    return avg_loss, acc

# --- 6. PHASE 1: WARMUP (Classifier Only) ---
print(f"\n=== PHASE 1: WARMUP ({WARMUP_EPOCHS} Epochs) ===")
print("Only training the classifier head. Feature extractor is frozen.")

# Optimizer for Phase 1: Only optimize the classifier parameters
optimizer_warmup = optim.Adam(model.classifier.parameters(), lr=1e-3)

for epoch in range(WARMUP_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_warmup, "Warmup")
    val_loss, val_acc = validate(model, val_loader)
    
    print(f"Epoch [{epoch+1}/{WARMUP_EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

print("\n--> Phase 1 Complete. Model head is initialized.")

# --- 7. PHASE 2: FINE-TUNING (Partial Unfreeze) ---
print(f"\n=== PHASE 2: FINE-TUNING ({FINETUNE_EPOCHS} Epochs) ===")
print("Unfreezing last 2 blocks of EfficientNet. Using differential learning rates.")

# A. Unfreeze the last 2 feature blocks (EfficientNet specific)
# model.features is a sequence of blocks. We unfreeze the last few.
for param in model.features[-2:].parameters():
    param.requires_grad = True

# B. Differential Learning Rate:
# - Low LR (1e-4) for the backbone (to preserve ImageNet features)
# - High LR (1e-3) for the classifier (to keep learning fast)
optimizer_finetune = optim.Adam([
    {'params': model.features[-2:].parameters(), 'lr': 1e-4}, # Backbone
    {'params': model.classifier.parameters(), 'lr': 1e-3}     # Head
], weight_decay=1e-4) # Added weight decay for regularization

# Scheduler for Phase 2
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_finetune, mode='max', factor=0.1, patience=2)

best_acc = 0.0

for epoch in range(FINETUNE_EPOCHS):
    current_epoch = epoch + 1 + WARMUP_EPOCHS
    
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_finetune, "FineTune")
    val_loss, val_acc = validate(model, val_loader)
    
    scheduler.step(val_acc)
    
    print(f"Epoch [{current_epoch}/{TOTAL_EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "deepdetect_robust_best.pth")
        print(f"--> Best model saved! ({best_acc:.2f}%)")

print("\nFull training complete.")

# --- 1. CONFIGURATION ---
DATASET_NAME = "ayushmandatta1/deepdetect-2025" 
DOWNLOAD_ROOT = "./deepdetect_data"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 32
WARMUP_EPOCHS = 5      # Phase 1: Train classifier only
FINETUNE_EPOCHS = 10   # Phase 2: Train backbone + classifier
TOTAL_EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS

# --- 2. STRONG AUGMENTATION (REGULARIZATION) ---
# Deepfake detection requires robustness to quality loss and slight geometry changes.
train_transform = transforms.Compose([
    transforms.Resize((260, 260)), # B2 Input Size
    
    # Geometric Augmentations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Simulate slight camera shifts
    
    # Quality/Color Augmentations (Crucial for Deepfakes)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Simulate blur/compression artifacts
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((260, 260)), # B2 Input Size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=train_transform)
val_dataset = datasets.ImageFolder(root=VAL_PATH, transform=val_transform)

# num_workers=0 avoids Windows/VSCode freeze issues
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Classes: {train_dataset.classes}")

# --- 4. MODEL SETUP ---
print("Loading EfficientNet-B2...")
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)

# A. INITIAL FREEZE: Freeze EVERYTHING first
for param in model.parameters():
    param.requires_grad = False

# B. MODIFY CLASSIFIER: High Dropout as requested (0.4 - 0.5)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5), # Stronger dropout
    nn.Linear(num_features, 2)
)

model = model.to(DEVICE)

# --- 5. LOSS FUNCTION (With Label Smoothing) ---
# Label smoothing prevents the model from being "too confident" (overfitting)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# --- HELPER FUNCTION FOR TRAINING ---
def train_one_epoch(model, loader, optimizer, phase_name):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    acc = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, acc

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels) # Calculate val loss too
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    avg_loss = val_loss / len(loader)
    return avg_loss, acc

# --- 6. PHASE 1: WARMUP (Classifier Only) ---
print(f"\n=== PHASE 1: WARMUP ({WARMUP_EPOCHS} Epochs) ===")
print("Only training the classifier head. Feature extractor is frozen.")

# Optimizer for Phase 1: Only optimize the classifier parameters
optimizer_warmup = optim.Adam(model.classifier.parameters(), lr=1e-3)

# History Tracking
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(WARMUP_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_warmup, "Warmup")
    val_loss, val_acc = validate(model, val_loader)
    
    # Update History
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{WARMUP_EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

print("\n--> Phase 1 Complete. Model head is initialized.")

# --- 7. PHASE 2: FINE-TUNING (Partial Unfreeze) ---
print(f"\n=== PHASE 2: FINE-TUNING ({FINETUNE_EPOCHS} Epochs) ===")
print("Unfreezing last 2 blocks of EfficientNet. Using differential learning rates.")

# A. Unfreeze the last 2 feature blocks (EfficientNet specific)
# model.features is a sequence of blocks. We unfreeze the last few.
for param in model.features[-2:].parameters():
    param.requires_grad = True

# B. Differential Learning Rate:
# - Low LR (1e-4) for the backbone (to preserve ImageNet features)
# - High LR (1e-3) for the classifier (to keep learning fast)
optimizer_finetune = optim.Adam([
    {'params': model.features[-2:].parameters(), 'lr': 1e-4}, # Backbone
    {'params': model.classifier.parameters(), 'lr': 1e-3}     # Head
], weight_decay=1e-4) # Added weight decay for regularization

# Scheduler for Phase 2
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_finetune, mode='max', factor=0.1, patience=2)

best_acc = 0.0
patience_counter = 0
EARLY_STOPPING_PATIENCE = 3

for epoch in range(FINETUNE_EPOCHS):
    current_epoch = epoch + 1 + WARMUP_EPOCHS
    
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_finetune, "FineTune")
    val_loss, val_acc = validate(model, val_loader)
    
    # Update History
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
    
    scheduler.step(val_acc)
    
    print(f"Epoch [{current_epoch}/{TOTAL_EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "deepdetect_robust_best.pth")
        print(f"--> Best model saved! ({best_acc:.2f}%)")
    else:
        patience_counter += 1
        print(f"--> Early Stopping Counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

print("\nFull training complete.")

# --- 8. PLOTTING ---
try:
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.title('Accuracy History (Robust B2)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Loss History (Robust B2)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig('training_curves_robust_b2.png')
    print("Curves saved to training_curves_robust_b2.png")
except Exception as e:
    print(f"Error plotting curves: {e}")