from collections import defaultdict
import random
from typing import Tuple, Optional

from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as T

from pathlib import Path

class DownsampledTensorDataset(Dataset):
    """
    Loads preprocessed tensors from a directory structure:
    - root_dir/
        - no_title/
            - image1.pt
            - image2.pt
            ...
        - title/
            - image1.pt
            - image2.pt
            ...
    """
    def __init__(self, root_dir: str, transform=None):
        self.samples = []  # list of (tensor_path, label)
        self.class_to_idx = {"no_title": 0, "title": 1}
        self.transform = transform

        root_dir = Path(root_dir)
        for class_name, label in self.class_to_idx.items():
            for pt_path in (root_dir / class_name).glob("*.pt"):
                self.samples.append((pt_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, label = self.samples[idx]
        tensor = torch.load(pt_path)  # shape: [1, H, W], already in [-1, 1]

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label

def compute_antialiasing_sigma(orig_size, target_size):
    """
    Computes the sigma for Gaussian blur to apply anti-aliasing when resizing images.
    :param orig_size: Tuple[int, int] - original image size (width, height)
    :param target_size: Tuple[int, int] - target size (width, height)
    :return: float - sigma value for Gaussian blur
    """
    scale = max(orig_size[0] / target_size[0], orig_size[1] / target_size[1])
    return 0.5 * scale if scale > 1.0 else 0.0

class LazyCacheDataset(Dataset):
    """
    A dataset that lazily loads images from disk, applies anti-aliasing and resizing,
    and caches the results in memory to avoid repeated disk I/O.
    Directory structure:
    - root_dir/
        - no_title/
            - image1.jpg
            - image2.png
            ...
        - title/
            - image1.jpg
            - image2.png
            ...
    """
    def __init__(self, root_dir, target_size=(64, 32)):
        self.target_size = target_size
        self.class_to_idx = {"no_title": 0, "title": 1}
        self.samples = []  # List[Tuple[Path, int]]
        self.cache = {}    # Dict[int, Tuple[Tensor, int]]

        for class_name in ("no_title", "title"):
            label = self.class_to_idx[class_name]
            for img_path in Path(root_dir, class_name).glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("L")
        except:
            print(f"Error loading image {path}. Skipping.")
            return None
        sigma = compute_antialiasing_sigma(img.size, self.target_size)
        if sigma > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        img = img.resize(self.target_size, resample=Image.Resampling.BILINEAR)
        tensor = T.ToTensor()(img)
        tensor = (tensor - tensor.min()) * 2 / (tensor.max() - tensor.min()) - 1

        self.cache[idx] = (tensor, label)
        return self.cache[idx]

def get_train_val_dataloaders(
    root_dir: str,
    target_size: Tuple[int, int] = (64, 32),
    batch_size: int = 64,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders from a directory of images.
    :param root_dir: Path to the root directory containing images.
    :param target_size: Tuple[int, int] - target size for resizing images.
    :param batch_size: int - batch size for DataLoader.
    :param val_fraction: float - fraction of data to use for validation.
    :param seed: int - random seed for reproducibility.
    :return: tuple of (train_loader, val_loader)
    """
    dataset = LazyCacheDataset(root_dir, target_size)
    total = len(dataset)
    val_len = int(val_fraction * total)
    train_len = total - val_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def split_dataset_balanced(dataset: Dataset, val_size: float = 0.1, test_size: float = 0.1, seed: int = 42) -> Tuple[Subset, Subset, Subset]:
    """
    Splits a dataset into train, validation, and test sets while maintaining class balance.
    :param dataset: Dataset - the dataset to split.
    :param val_size: float - fraction of data to use for validation.
    :param test_size: float - fraction of data to use for testing.
    :param seed: int - random seed for reproducibility.
    :return: tuple of (train_set, val_set, test_set)
    """
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    random.seed(seed)
    train_indices, val_indices, test_indices = [], [], []

    for label, indices in class_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_val = int(n * val_size)
        n_test = int(n * test_size)

        val_indices += indices[:n_val]
        test_indices += indices[n_val:n_val+n_test]
        train_indices += indices[n_val+n_test:]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def get_balanced_tensor_datasets(
    root_dir: str,
    batch_size: int = 64,
    val_size: float = 0.2,
    test_size: float = 0.1,
    flip_augment: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test dataloaders from a preprocessed tensor dataset.
    :param root_dir: str - path to the root directory containing preprocessed tensors.
    :param batch_size: int - batch size for DataLoader.
    :param val_size: float - fraction of data to use for validation.
    :param test_size: float - fraction of data to use for testing.
    :param flip_augment: bool - whether to apply random horizontal and vertical flips for data augmentation.
    :return: tuple of (train_loader, val_loader, test_loader)
    """
    dataset = DownsampledTensorDataset(root_dir)  # load preprocessed dataset
    train_set, val_set, test_set = split_dataset_balanced(dataset, val_size=val_size, test_size=test_size)

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    if flip_augment:
        train_loader.dataset.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

    return train_loader, val_loader, test_loader