"""
DataModule for the Chesapeake Bay dataset for segmentation tasks.

This implementation provides a structured way to handle the data loading and
preprocessing required for training and validating a segmentation model.

Dataset citation:
Robinson C, Hou L, Malkin K, Soobitsky R, Czawlytko J, Dilkina B, Jojic N.
Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data.
Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition
(CVPR 2019).

Dataset URL: https://lila.science/datasets/chesapeakelandcover
"""

import re
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChesapeakeDataset(Dataset):
    """
    Dataset class for the Chesapeake Bay segmentation dataset.

    Args:
        chip_dir (str): Directory containing the image chips.
        label_dir (str): Directory containing the labels.
        metadata (Box): Metadata for normalization and other dataset-specific details.
        platform (str): Platform identifier used in metadata.
        num_classes (int): Number of classes for segmentation.
    """

    def __init__(self, chip_dir, label_dir, metadata, platform, num_classes=None, is_training=False):
        self.chip_dir = Path(chip_dir)
        self.label_dir = Path(label_dir)
        self.metadata = metadata
        self.num_classes = num_classes
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )
        self.is_training = is_training
        self.augmentation = self.create_augmentation() if is_training else None

        # Load chip and label file names
        self.chips = [chip_path.name for chip_path in self.chip_dir.glob("*.npy")][
            :1000
        ]
        # self.labels = [re.sub("_naip-new_", "_lc_", chip) for chip in self.chips]
        self.labels = [re.sub("_stack_", "_crop_label_", chip) for chip in self.chips]
        
        if is_training:
            self.sample_non_empty_chips(keep_empty_ratio=0.25)
        
    def create_transforms(self, mean, std):
        """
        Create normalization transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        return v2.Compose(
            [
                v2.Normalize(mean=mean, std=std),
            ],
        )
        
    def create_augmentation(self):
        """
        Create augmentation transforms.

        Returns:
            albumentations.Compose: A composition of augmentation transforms.
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.Affine(
            #     translate_percent=0.1,  # equivalent to shift_limit
            #     scale=(0.8, 1.2),       # equivalent to scale_limit
            #     rotate=(-15, 15),
            #     mode=0,                 # fill value for image
            #     cval=0,                 # same as mode=0
            #     mask_value=0,             # fill value for mask
            #     p=0.3,
            # )
        ])
        
    def sample_non_empty_chips(self, keep_empty_ratio=0.5):
        """
        Sample non-empty chips from the dataset.
        """
        valid_chips, valid_labels = [], []
        num_empty, num_nonempty = 0, 0

        for chip_name, label_name in zip(self.chips, self.labels):
            label_path = self.label_dir / label_name
            if not label_path.exists():
                continue

            label = np.load(label_path, mmap_mode="r")  # faster than loading into RAM fully

            if np.any(label > 0):
                num_nonempty += 1
                valid_chips.append(chip_name)
                valid_labels.append(label_name)
            else:
                num_empty += 1
                if np.random.rand() < keep_empty_ratio:
                    valid_chips.append(chip_name)
                    valid_labels.append(label_name)

        print(
            f"[INFO] Filtered dataset: {len(valid_chips)} / {len(self.chips)} kept "
            f"({100 * len(valid_chips) / len(self.chips):.1f}%). "
            f"Non-empty: {num_nonempty}, Empty: {num_empty}, "
            f"Kept {keep_empty_ratio*100:.0f}% of empty."
        )

        self.chips = valid_chips
        self.labels = valid_labels

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the image, label, and additional information.
        """
        chip_name = self.chip_dir / self.chips[idx]
        label_name = self.label_dir / self.labels[idx]

        chip = np.load(chip_name).astype(np.float32)
        label = np.load(label_name)
        
        if self.num_classes is not None and self.num_classes > 2:
            label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 15: 6}
            remapped_label = np.copy(label)
            for src, dst in label_mapping.items():
                remapped_label[label == src] = dst
        else:
            # Binary case (values already 0 and 1)
            remapped_label = label

        # Remove single-band dimension if present
        if remapped_label.ndim == 3 and remapped_label.shape[0] == 1:
            remapped_label = remapped_label[0]


        # # Remap labels to match desired classes
        # label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 15: 6}
        # remapped_label = np.vectorize(label_mapping.get)(label)

        # Augmentation
        if self.is_training:
            chip = np.transpose(chip, (1, 2, 0))
            
            augmented = self.augmentation(image=chip, mask=remapped_label)
            chip_aug = augmented["image"]  # (H, W, C)
            label_aug = augmented["mask"]  # (H, W)

            chip_tensor = torch.from_numpy(np.transpose(chip_aug, (2, 0, 1))).float()  # (C, H, W)
            remapped_label = torch.from_numpy(label_aug) # (H, W)

            pixels = self.transform(chip_tensor)
            
        else:
            # No Augmentation
            pixels = self.transform(torch.from_numpy(chip))
            remapped_label = torch.from_numpy(remapped_label)
                        
        sample = {
            "pixels": pixels,
            "label": remapped_label,
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for latlon information
        }
        return sample


class ChesapeakeDataModule(L.LightningDataModule):
    """
    DataModule class for the Chesapeake Bay dataset.

    Args:
        train_chip_dir (str): Directory containing training image chips.
        train_label_dir (str): Directory containing training labels.
        val_chip_dir (str): Directory containing validation image chips.
        val_label_dir (str): Directory containing validation labels.
        metadata_path (str): Path to the metadata file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        platform (str): Platform identifier used in metadata.
        num_classes (int): Number of classes for segmentation.
    """

    def __init__(  # noqa: PLR0913
        self,
        train_chip_dir,
        train_label_dir,
        val_chip_dir,
        val_label_dir,
        metadata_path,
        batch_size,
        num_workers,
        platform,
        num_classes=None,
    ):
        super().__init__()
        self.train_chip_dir = train_chip_dir
        self.train_label_dir = train_label_dir
        self.val_chip_dir = val_chip_dir
        self.val_label_dir = val_label_dir
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.platform = platform
        self.num_classes = num_classes

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('fit' or 'test').
        """
        if stage in {"fit", None}:
            self.trn_ds = ChesapeakeDataset(
                self.train_chip_dir,
                self.train_label_dir,
                self.metadata,
                self.platform,
                self.num_classes,
                is_training=True,
            )
            self.val_ds = ChesapeakeDataset(
                self.val_chip_dir,
                self.val_label_dir,
                self.metadata,
                self.platform,
                self.num_classes,
                is_training=False,
            )

    def train_dataloader(self):
        """
        Create DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training dataset.
        """
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Create DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
