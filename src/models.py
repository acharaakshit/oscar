import torch.nn as nn
import timm_3d
import timm
import lightning as L
import torch
import torch.nn.functional as F
import torch.optim as optim
from monai.networks.nets.swin_unetr import SwinTransformer, PatchMerging, PatchMergingV2
import numpy as np
from collections.abc import Sequence
from torch.nn import LayerNorm
from monai.utils import ensure_tuple_rep, look_up_option
from monai.networks.nets import vit
from torchmetrics import F1Score
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

class Classifier2D(L.LightningModule):
    def __init__(self, 
                model_alias: str, 
                num_classes: int,
                lr: float = 1e-4,
                in_channels: int = 3,
                img_size = 128,
                use_rcs_consistency: bool = False,
                rcs_mask: torch.Tensor | None = None,
                up: float = 5,
                down: float = 2,
                random_shuffle: bool = False,
                shuffle_seed: int = 0,
                ):
        super().__init__()

        self.model_alias = model_alias
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = img_size
        self.use_rcs_consistency = use_rcs_consistency
        self._rcs_mask_tensor = rcs_mask
        self.up = up
        self.down = down
        self.shuffle = random_shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_gen = None  # lazily created per device in forward

        if self.use_rcs_consistency:
            assert self._rcs_mask_tensor is not None, "mask should be provided in this case"
            m = self._rcs_mask_tensor
            assert m.ndim == 4, "expect a 4D mask"
            self.register_buffer("rcs_img_mask", m, persistent=True)

        if model_alias.lower() == "vit":
            if in_channels != 3:
                raise ValueError("torchvision ViT-B/16 expects 3-channel input.")
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            # build ViT-B/16 from torchvision
            self.model = vit_b_16(weights=weights, image_size=img_size, dropout=0.1)
            in_feats = self.model.heads.head.in_features
            if self.model.heads.head.out_features != num_classes:
                self.model.heads.head = nn.Linear(in_feats, num_classes)
    
        elif 'swin' in model_alias.lower():
            self.model = timm.create_model(
                model_alias,
                num_classes=self.num_classes,
                in_chans=self.in_channels,
                img_size=self.img_size,
                drop_rate=0.2,
                pretrained=True
            )
        else:
            self.model = timm.create_model(
                model_alias,
                num_classes=self.num_classes,
                in_chans=self.in_channels,
                drop_rate=0.2,
                pretrained=True,
            )
        

        self.fscore = F1Score(task='multiclass', average='macro', num_classes=self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self.lr = lr
        self.in_channels = in_channels

    def forward(self, x):
        if not self.use_rcs_consistency:
            return self.model(x)

        f = self.model.forward_features(x)
        if f.ndim != 4: 
            raise RuntimeError(f"Expected 4D features from forward_features, got {f.shape}")

        B, C, Hf, Wf = f.shape
        m = self.rcs_img_mask.to(dtype=f.dtype, device=f.device)

        if self.shuffle:
            if self.shuffle_gen is None or self.shuffle_gen.device != x.device:
                self.shuffle_gen = torch.Generator(device=x.device).manual_seed(self.shuffle_seed)
            rcs_mitig = self.rcs_img_mask
            H, W = rcs_mitig.shape[-2:]
            # sample shifts from the far halves only
            dy = int(torch.randint(low=H//2, high=H, size=(1,), device=rcs_mitig.device, generator=self.shuffle_gen))
            dx = int(torch.randint(low=W//2, high=W, size=(1,), device=rcs_mitig.device, generator=self.shuffle_gen))
            m  = torch.roll(rcs_mitig, shifts=(dy, dx), dims=(-2, -1))


        if m.shape[-2:] != (Hf, Wf):
            m = F.adaptive_avg_pool2d(m, (Hf, Wf))
        

        w = m.expand(B, 1, Hf, Wf) 


        amax_b = w.view(B, -1).abs().amax(dim=1).view(B, 1, 1, 1) + 1e-8
        m_norm = w / amax_b
        scale = torch.where(m_norm < 0, self.up, self.down)
        w = 1.0 - m_norm * scale
        pooled = (f * w).sum((2, 3)) / (w.sum((2, 3)) + 1e-8)

        return self.model.fc(pooled) 


    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs.to(torch.bfloat16))
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs.to(torch.bfloat16))
        loss = self.loss_fn(outputs, labels)
        self.fscore(outputs, labels)
        self.log("val_fscore", self.fscore, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1, verbose=True
        )
        return {"optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.forward(inputs.to(torch.bfloat16))
        if self.num_classes > 1:
            preds = torch.argmax(preds, dim=1)
        return preds, labels

class CNN3DClassifier(L.LightningModule):
    def __init__(self, 
                model_alias: str, 
                num_classes: int,
                lr: float = 1e-4,
                in_channels: int = 1,
                pretrained: bool = True,
                drop_rate: float = 0.0,
                ):
        super().__init__()

        self.model_alias = model_alias
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.drop_rate = drop_rate
        
        self.model = timm_3d.create_model(
            model_alias,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            in_chans=self.in_channels,
            drop_rate=self.drop_rate,
        )

        self.fscore = F1Score(task='multiclass', average='macro', num_classes=self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self.lr = lr
        self.in_channels = in_channels

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs.to(torch.bfloat16))
        loss = self.loss_fn(outputs, labels)

        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs.to(torch.bfloat16))
        loss = self.loss_fn(outputs, labels)
        self.fscore(outputs, labels)
        self.log("val_fscore", self.fscore, prog_bar=True, logger=True, sync_dist=True)

        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=self.lr/50
        )
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.forward(inputs.to(torch.bfloat16))
        if self.num_classes > 1:
            preds = torch.argmax(preds, dim=1)
        return preds, labels

## UNDER DEVELOPMENT -- NOT MEANT FOR USE CURRENTLY
class VisionTransformer3DClassifier(L.LightningModule):
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 img_size: Sequence[int] | int = (256,256,256),
                 patch_size: Sequence[int] | int = (16,16,16), # reduce path size to check performance
                 hidden_size: int = 768,
                 drop_rate: float = 0.0,
                 lr: float = 1e-4,
                ):
        
            super().__init__()
            self.in_channels = in_channels
            self.img_size = img_size
            self.num_classes = num_classes
            self.patch_size = patch_size
            self.hidden_size = hidden_size
            self.dropout_rate = drop_rate

            self.model = vit.ViT(
                            in_channels=self.in_channels,
                            img_size=self.img_size,
                            num_classes=self.num_classes,
                            patch_size=self.patch_size,
                            classification=False,
                            post_activation=None,
                            dropout_rate=self.dropout_rate
                        )
            if self.num_classes == 1:
                self.fc = nn.Sequential(
                        nn.Linear(self.hidden_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, self.num_classes)
                    )
                self.loss_fn = nn.MSELoss()
            else:
                self.fc = nn.Linear(self.hidden_size, self.num_classes)
                self.loss_fn = nn.CrossEntropyLoss()
                self.fscore = F1Score(task='multiclass', average='macro', num_classes=self.num_classes)

            self.lr = lr
            
    def forward(self, x):
        x = self.model(x)
        x = x[0][:, 0]
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs.to(torch.bfloat16))
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs.to(torch.bfloat16))
        loss = self.loss_fn(outputs, labels)
        fscore = self.fscore(outputs, labels)
        self.log("val_fscore", fscore, prog_bar=True, logger=True, sync_dist=True)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=self.lr/100
        )
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.forward(inputs.to(torch.bfloat16))
        if self.num_classes > 1:
            preds = torch.argmax(preds, dim=1)
        return preds, labels

class SwinTransformer3DClassifier(L.LightningModule):
    def __init__(self, 
                in_channels: int,
                num_classes:int,
                img_size: Sequence[int] | int = (96,96,96), # just needs to be divisible by 2**5
                patch_size: int = 2,
                depths: Sequence[int] = (2, 2, 2, 2),
                num_heads: Sequence[int] = (3, 6, 12, 24),
                window_size: Sequence[int] | int = 5,
                qkv_bias: bool = True,
                mlp_ratio: float = 4.0,
                feature_size: int = 24,
                norm_name: tuple | str = "instance",
                drop_rate: float = 0,
                attn_drop_rate: float = 0.0,
                dropout_path_rate: float = 0.0,
                normalize: bool = True,
                norm_layer: type[LayerNorm] = nn.LayerNorm,
                patch_norm: bool = False,
                use_checkpoint: bool = False,
                spatial_dims: int = 3,
                downsample: str | nn.Module = "merging",
                use_v2: bool = True,
                lr: float = 1e-4,
                ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self.patch_size = patch_size

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")
        
        MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

        self.swin_transformer = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )

        # Classification head
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # Global Pooling
        self.num_classes = num_classes
        self.fc = nn.Linear(feature_size * 16, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.fscore = F1Score(task='multiclass', average='macro', num_classes=self.num_classes)
        
        self.lr = lr

    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )
    
    def forward(self, x):
            x = self.swin_transformer(x)
            x = x[-1]
            x = self.global_avg_pool(x).flatten(1)
            x = self.fc(x)
            return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs.to(torch.bfloat16))
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs.to(torch.bfloat16))
        if self.num_classes == 1: #regression
            loss = self.loss_fn(outputs, labels.view(-1, 1))
        else:
            loss = self.loss_fn(outputs, labels)
            fscore = self.fscore(outputs, labels)
            self.log("val_fscore", fscore, prog_bar=True, logger=True, sync_dist=True)            

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.forward(inputs.to(torch.bfloat16))
        if self.num_classes > 1:
            preds = torch.argmax(preds, dim=1)
        return preds, labels

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=self.lr/100
        )
        return [optimizer], [scheduler]
