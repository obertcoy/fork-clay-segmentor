from typing import Literal

import lightning as L
import torch
import yaml
from box import Box

from claymodel.model import clay_mae_base, clay_mae_large, clay_mae_small, clay_mae_tiny


class ClayMAEModule(L.LightningModule):
    def __init__(  # noqa: PLR0913
        self,
        model_size="base",
        mask_ratio=0.75,
        norm_pix_loss=False,
        patch_size=8,
        shuffle=False,
        metadata_path="configs/metadata.yaml",
        teacher="samvit_base_patch16.sa1b",
        dolls=[16, 32, 64, 128, 256, 768],
        doll_weights=[1, 1, 1, 1, 1, 1],
        lr=1e-5,
        wd=0.05,
        b1=0.9,
        b2=0.95,
        embeddings_level: Literal["mean", "patch", "group"] = "mean",
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        model_map = {
            "tiny": clay_mae_tiny,
            "small": clay_mae_small,
            "base": clay_mae_base,
            "large": clay_mae_large,
        }
        if model_size in model_map:
            model_args = {
                "mask_ratio": mask_ratio,
                "patch_size": patch_size,
                "norm_pix_loss": norm_pix_loss,
                "shuffle": shuffle,
                "metadata": self.metadata,
                "teacher": teacher,
                "dolls": dolls,
                "doll_weights": doll_weights,
            }
            self.model = model_map[model_size](**model_args)
            # checkpoint_path = 'mae_v1.5.0_epoch-76_val-loss-0.1612.ckpt'
            # checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # # Extract the state dictionary
            # state_dict = checkpoint['state_dict']

            # # Modify the state dictionary
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     # Remove 'model.' prefix if it exists
            #     if k.startswith('model.'):
            #         k = k[len('model.'):]
            #     # Exclude keys related to the 'teacher'
            #     if not (k.startswith('teacher') or k.startswith('mrl')):
            #         new_state_dict[k] = v
            # with torch.no_grad():
            #     # Load the modified state dictionary into your model
            #     missing_keys, unexpected_keys = (
            #         self.model.load_state_dict(new_state_dict, strict=False)
            #     )
            #     # Optionally, print missing and unexpected keys
            #     print(f"Missing keys: {missing_keys}")
            #     print(f"Unexpected keys: {unexpected_keys}")
        else:
            raise ValueError(
                f"Invalid model size {model_size}. Expected one of {model_map.keys()}"
            )

    def on_train_epoch_start(self):
        self.model.teacher.eval()

    def forward(self, datacube: dict[str, torch.Tensor]):
        return self.model(datacube)

    def configure_optimizers(self):
        """
        Configure optimizer with layer-specific learning rates.
        """
        # Group parameters by layer type
        decoder_params = []
        encoder_layers_23_24 = []  # Last 2 layers
        encoder_layers_21_22 = []  # Second-to-last 2 layers
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'encoder' in name:
                if 'transformer.layers.22' in name or 'transformer.layers.23' in name:
                    encoder_layers_23_24.append(param)
                elif 'transformer.layers.20' in name or 'transformer.layers.21' in name:
                    encoder_layers_21_22.append(param)
            else:
                decoder_params.append(param)
        
        # Different learning rates for different groups
        # Typically: decoder > encoder_late > encoder_early
        param_groups = [
            {
                'params': decoder_params,
                'lr': self.hparams.lr,  # Full LR for decoder
                'weight_decay': self.hparams.wd,
            },
            {
                'params': encoder_layers_23_24,
                'lr': self.hparams.lr / 5,  # Higher LR for last layers
                'weight_decay': self.hparams.wd,
            },
            {
                'params': encoder_layers_21_22,
                'lr': self.hparams.lr / 10,  # Lower LR for earlier unfrozen layers
                'weight_decay': self.hparams.wd,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=1,
            eta_min=self.hparams.lr * 100,  # FIXED!
            last_epoch=-1,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(self, batch: dict[str, torch.Tensor], batch_idx: int, phase: str):
        platform = batch["platform"][0]
        loss, reconstruction_loss, representation_loss = self(batch)

        losses = {
            "loss": loss,
            "rec_loss": reconstruction_loss,
            "rep_loss": representation_loss,
        }

        for loss_name, loss_value in losses.items():
            self.log(
                name=f"{phase}/{loss_name}",
                value=loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                name=f"{phase}_{platform}/{loss_name}",
                value=loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")
