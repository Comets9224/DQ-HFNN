"""
JAFFE Dataset Loader (Fully Reproducing Paper Settings)
Japanese Female Facial Expression (JAFFE) Database

Dataset Information:
- 10 Japanese female subjects
- 7 expressions: 0=neutral, 1=happy, 2=sad, 3=surprise, 4=anger, 5=disgust, 6=fear
- Total 213 images (original size 256×256)

Paper Preprocessing Steps:
1. ✅ Convert to grayscale
2. ✅ Extract facial expression region (face detection + cropping)
3. ✅ Resize to 32×32 (matches model input)
4. ✅ Data augmentation: random horizontal flip + random rotation
5. ✅ MNIST standard normalization (mean=0.5, std=0.5)
6. ✅ 8-fold cross validation

Usage Instructions:
- By default uses load_dataset_jaffe() function (simple split version)
- Uncomment corresponding function definition to use K-fold cross validation
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold


def extract_face(image_path, target_size=150, margin=20):
    """
    Extract face region from original image

    Parameters:
        image_path: Path to image file
        target_size: Target size for cropping (square)
        margin: Margin to expand around detected face

    Returns:
        face_img: Face region as PIL Image (grayscale)
    """
    # Read image (grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # Load OpenCV pre-trained face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # 如果检测到人脸，使用第一个（通常是最大的）
    if len(faces) > 0:
        x, y, w, h = faces[0]

        # Expand boundaries (add margin)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)

        # Crop the face region
        face = img[y:y+h, x:x+w]

        # Convert to square (using the longer side)
        size = max(w, h)
        square_face = np.zeros((size, size), dtype=np.uint8)

        # Center the content
        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        square_face[offset_y:offset_y+h, offset_x:offset_x+w] = face

        # Resize to target dimensions
        face_resized = cv2.resize(square_face, (target_size, target_size))

    else:
        # If no face is detected, use center crop as fallback
        print(f"⚠️ 未检测到人脸，使用中心裁剪: {os.path.basename(image_path)}")
        h, w = img.shape
        size = min(h, w)

        # Center cropping
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        center_crop = img[start_y:start_y+size, start_x:start_x+size]

        # Resize to target dimensions
        face_resized = cv2.resize(center_crop, (target_size, target_size))

    # Convert to PIL Image
    face_img = Image.fromarray(face_resized)

    return face_img


class JAFFEDataset(Dataset):
    """JAFFE Dataset Class (with face extraction)"""

    def __init__(self, image_paths, labels, transform=None, extract_face_region=True):
        """
        Parameters:
            image_paths: List of image paths
            labels: List of labels
            transform: Image transformations
            extract_face_region: Whether to extract face region (default True)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.extract_face_region = extract_face_region

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image paths
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Extract face region (key step from paper)
        if self.extract_face_region:
            image = extract_face(img_path, target_size=150)
        else:
            image = Image.open(img_path).convert('L')

        # Apply transformations (including resizing to 32×32 and normalization)
        if self.transform:
            image = self.transform(image)

        return image, label


def load_jaffe_raw(data_dir='./data/jaffe/data'):
    """
    Load raw JAFFE data

    Returns:
        image_paths: List of image paths
        labels: List of labels
        label_names: Dictionary of label names
    """
    # 7 expression categories
    label_names = {
        0: 'NE',  # Neutral (中性)
        1: 'HA',  # Happy (高兴)
        2: 'SA',  # Sad (悲伤)
        3: 'SU',  # Surprise (惊讶)
        4: 'AN',  # Angry (愤怒)
        5: 'DI',  # Disgust (厌恶)
        6: 'FE'  # Fear (恐惧)
    }

    # Mapping from expression codes to numerical labels
    emotion_to_label = {v: k for k, v in label_names.items()}

    image_paths = []
    labels = []

    # Scan data directory
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}\nPlease place JAFFE dataset in this directory")

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.tiff') or filename.endswith('.jpg') or filename.endswith('.png'):
            # JAFFE filename format: KA.HA1.30.tiff (person.expression.id.extension)
            parts = filename.split('.')
            if len(parts) >= 2:
                emotion_code = parts[1][:2]  # Extract emotion code (first two characters)

                if emotion_code in emotion_to_label:
                    image_paths.append(os.path.join(data_dir, filename))
                    labels.append(emotion_to_label[emotion_code])

    print(f"✅ Loaded JAFFE dataset: {len(image_paths)} images")
    print(f"   Label distribution: {np.bincount(labels)}")

    return image_paths, labels, label_names


def get_jaffe_transforms(train=True, image_size=32):
    """
    Get JAFFE data transformations (paper configuration)

    Key settings:
    1. image_size=32 matches model input requirements
    2. MNIST standard normalization (mean=0.5, std=0.5)

    Parameters:
        train: Whether for training set
        image_size: Target image size (default 32, matches model)
    """
    if train:
        # Training set: Data augmentation + normalization
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize to 32×32
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip (50% probability)
            transforms.RandomRotation(degrees=10),  # Random rotation (±10 degrees)
            transforms.ToTensor(),  # Convert to Tensor [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]
        ])
    else:
        # Test set: Only resize + normalization
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    return transform

def load_dataset_jaffe(k=8, fold=0,
                       data_dir='./data/jaffe/data',
                       image_size=32):
    """
    [K-fold Version] 8-Fold Cross Validation (paper standard configuration)

    Load JAFFE dataset (K-fold cross validation)
    Returns Dataset objects consistent with other datasets in run_test.py

    Parameters:
        k: Number of folds (paper uses k=8)
        fold: Current fold (0 to k-1)
        data_dir: Data directory
        image_size: Image size (default 32, matches model input)

    Returns:
        train_dataset: Training set Dataset object
        val_dataset: Validation set Dataset object
    """
    # Load raw data
    image_paths, labels, label_names = load_jaffe_raw(data_dir)

    # K-fold splitting
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Get current fold indices
    all_indices = np.arange(len(image_paths))
    splits = list(kf.split(all_indices))

    if fold >= k:
        raise ValueError(f"fold={fold} over range [0, {k - 1}]")

    train_idx, val_idx = splits[fold]

    # Split data
    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print("=" * 60)
    print(f"✅ JAFFE dataset (K-Fold={k}, Fold={fold})")
    print(f"   Training set: {len(train_paths)} samples")
    print(f"   Validation set: {len(val_paths)} samples")
    print(f"   Image size: {image_size}×{image_size}")
    print("=" * 60)

    # Create Dataset objects (return format matches CIFAR-10/DirtyMNIST)
    train_dataset = JAFFEDataset(
        train_paths,
        train_labels,
        transform=get_jaffe_transforms(train=True, image_size=image_size),
        extract_face_region=True  # Key step from paper: Extract face region
    )

    val_dataset = JAFFEDataset(
        val_paths,
        val_labels,
        transform=get_jaffe_transforms(train=False, image_size=image_size),
        extract_face_region=True  # Key step from paper: Extract face region
    )

    return train_dataset, val_dataset

