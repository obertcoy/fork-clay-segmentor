"""
LightningModule for training and validating a segmentation model using the
Segmentor class.
"""

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import F1Score, MulticlassJaccardIndex, BinaryJaccardIndex, BinaryF1Score
from box import Box
import yaml

from claymodel.finetune.segment.factory import Segmentor


class ChesapeakeSegmentor(L.LightningModule):
    """
    LightningModule for segmentation tasks, utilizing Clay Segmentor.

    Attributes:
        model (nn.Module): Clay Segmentor model.
        loss_fn (nn.Module): The loss function.
        iou (Metric): Intersection over Union metric.
        f1 (Metric): F1 Score metric.
        lr (float): Learning rate.
    """

    def __init__(  # # noqa: PLR0913
        self,
        num_classes,
        ckpt_path,
        lr,
        wd,
        b1,
        b2,
        metadata_path=None,
        platform=None,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing
        self.model = Segmentor(
            num_classes=num_classes,
            ckpt_path=ckpt_path,
        )
        
        # Load metadata if provided
        if metadata_path is not None:
            with open(metadata_path, 'r') as f:
                self.metadata = Box(yaml.safe_load(f))
        else:
            self.metadata = None
        
        self.platform = platform

        # Use binary or multiclass based on num_classes
        if num_classes == 2:
            # Binary segmentation
            self.loss_fn = smp.losses.FocalLoss(mode="binary", alpha=0.5)
            self.iou = BinaryJaccardIndex()
            self.f1 = BinaryF1Score()
        else:
            # Multiclass segmentation
            self.loss_fn = smp.losses.FocalLoss(mode="multiclass")
            self.iou = MulticlassJaccardIndex(
                num_classes=num_classes,
                average="macro",
            )
            self.f1 = F1Score(
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            )

    def forward(self, datacube):
        """
        Forward pass through the segmentation model.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The segmentation logits.
        """
        # Extract GSD and wavelengths from metadata if available
        if self.metadata is not None and self.platform is not None:
            platform_meta = self.metadata[self.platform]
            # Get wavelengths in band_order sequence
            waves = torch.tensor([
                platform_meta.bands.wavelength[band] 
                for band in platform_meta.band_order
            ])
            gsd = torch.tensor(platform_meta.gsd)
        else:
            # Fallback to NAIP defaults (for backward compatibility)
            waves = torch.tensor([0.65, 0.56, 0.48, 0.842])  # NAIP wavelengths
            gsd = torch.tensor(1.0)  # NAIP GSD

        # Forward pass through the network
        return self.model(
            {
                "pixels": datacube["pixels"],
                "time": datacube["time"],
                "latlon": datacube["latlon"],
                "gsd": gsd,
                "waves": waves,
            },
        )

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler
            configuration.
        """
        optimizer = optim.AdamW(
            [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=1,
            eta_min=self.hparams.lr * 100,
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training and validation.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.
            phase (str): The phase (train or val).

        Returns:
            torch.Tensor: The loss value.
        """
        labels = batch["label"].long()
        outputs = self(batch)
        outputs = F.interpolate(
            outputs,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )  # Resize to match labels size

        # For binary segmentation, use only the positive class channel (channel 1)
        # Binary loss expects [B, 1, H, W] shape
        if self.hparams.num_classes == 2:
            outputs_binary = outputs[:, 1:2, :, :].contiguous()  # Take only channel 1 (positive class) and make contiguous
            loss = self.loss_fn(outputs_binary, labels)
            # Apply sigmoid to convert logits to probabilities, then squeeze to [B, H, W] for metrics
            outputs_probs = torch.sigmoid(outputs_binary).squeeze(1)  # [B, H, W]
            iou = self.iou(outputs_probs, labels)
            f1 = self.f1(outputs_probs, labels)
        else:
            # Multiclass case
            loss = self.loss_fn(outputs, labels)
            iou = self.iou(outputs, labels)
            f1 = self.f1(outputs, labels)

        # Log metrics
        self.log(
            f"{phase}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/iou",
            iou,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/f1",
            f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "val")
