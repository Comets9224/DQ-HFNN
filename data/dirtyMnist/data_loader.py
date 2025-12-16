import torch
import ddu_dirty_mnist
from torch.utils.data import Subset, Dataset
import numpy as np


class NormalizedDataset(Dataset):
    """
    Dataset normalization wrapper
    Applies standard MNIST normalization to tensors returned by ddu_dirty_mnist
    """

    def __init__(self, base_dataset, mean=0.1307, std=0.3081):
        """
        Parameters:
            base_dataset: Original dataset
            mean: Standard MNIST mean
            std: Standard MNIST standard deviation
        """
        self.base_dataset = base_dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):

        image, label = self.base_dataset[idx]


        image = (image - self.mean) / self.std

        return image, label


def load_dataset_dmnist(train_transform=None, test_transform=None, use_paper_version=True):
    """
    Load Dirty MNIST dataset (paper standard settings + MNIST normalization)

    1. ✅ MNIST standard normalization (mean=0.1307, std=0.3081)
    2. ❌ No data augmentation (no rotation, translation, scaling etc.)
    3. ✅ Sampled to 60,000 examples (paper standard setting)
    4. ✅ transform=None (raw image input, normalization handled by wrapper class)

    Parameters:
        train_transform: Training set transforms (default None, matches paper)
        test_transform: Test set transforms (default None, matches paper)
        use_paper_version: Whether to use paper version (60k samples), default True

    Returns:
        train_data, test_data

    Expected performance:
        - Normalized version: 86-88% (standard deep learning practice)
    """


    if train_transform is None:
        train_transform = None
        print("=" * 60)
        print("✅ Using paper configuration: transform=None (raw pixel values [0,1])")
        print("=" * 60)

    if test_transform is None:
        test_transform = None

    dirty_mnist_train_full = ddu_dirty_mnist.DirtyMNIST(
        "./data/dirtyMnist/data",
        train=True,
        transform=train_transform,
        download=True,
        device="cpu"
    )


    dirty_mnist_test = ddu_dirty_mnist.DirtyMNIST(
        "./data/dirtyMnist/data",
        train=False,
        transform=test_transform,
        download=True,
        device="cpu"
    )


    if use_paper_version:
        print("=" * 60)
        print("✅ Using paper version: Sampling 120k training examples down to 60k")
        print("=" * 60)

        total_samples = len(dirty_mnist_train_full)
        print(f"Original training set size: {total_samples}")

        # Strategy: Randomly sample 50% (maintaining the mix ratio of MNIST and AmbiguousMNIST)
        np.random.seed(42)
        indices = np.random.choice(total_samples, 60000, replace=False)
        indices = sorted(indices)

        dirty_mnist_train = Subset(dirty_mnist_train_full, indices)

        print(f"Sampled training set size: {len(dirty_mnist_train)}")
        print(f"Sampling ratio: {len(dirty_mnist_train) / total_samples:.1%}")
        print("=" * 60)

    else:
        # Use the full 120k samples
        print(f"Using full version: {len(dirty_mnist_train_full)} training samples")
        dirty_mnist_train = dirty_mnist_train_full

    dirty_mnist_train = NormalizedDataset(dirty_mnist_train)
    dirty_mnist_test = NormalizedDataset(dirty_mnist_test)

    return dirty_mnist_train, dirty_mnist_test