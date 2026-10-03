"""Versioned multi-task model builders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


SUPPORTED_BACKBONES = (
    "convnext_tiny",
    "efficientnet_v2_s",
    "mobilenet_v2",
    "dinov2_vits14",
)


@dataclass(frozen=True)
class ModelDimensions:
    species: int
    sex: int = 2
    mating: int = 2
    morph: int = 2


def _torchvision_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        feature_dim = int(backbone.classifier[-1].in_features)
        backbone.classifier[-1] = nn.Identity()
        return backbone, feature_dim
    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_v2_s(weights=weights)
        feature_dim = int(backbone.classifier[-1].in_features)
        backbone.classifier[-1] = nn.Identity()
        return backbone, feature_dim
    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        feature_dim = int(backbone.classifier[-1].in_features)
        backbone.classifier[-1] = nn.Identity()
        return backbone, feature_dim
    raise ValueError(f"Unsupported TorchVision backbone: {name}")


class MultiTaskClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        dimensions: ModelDimensions,
        *,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone {backbone_name!r}; choose from {SUPPORTED_BACKBONES}")
        self.backbone_name = backbone_name
        if backbone_name == "dinov2_vits14":
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained
            )
            feature_dim = 384
        else:
            self.backbone, feature_dim = _torchvision_backbone(backbone_name, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.species_head = nn.Linear(feature_dim, dimensions.species)
        self.sex_head = nn.Linear(feature_dim, dimensions.sex)
        self.mating_head = nn.Linear(feature_dim, dimensions.mating)
        self.morph_head = nn.Linear(feature_dim, dimensions.morph)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        if isinstance(features, dict):
            features = features.get("x_norm_clstoken", next(iter(features.values())))
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        features = self.dropout(features)
        return {
            "species_logits": self.species_head(features),
            "sex_logits": self.sex_head(features),
            "mating_logits": self.mating_head(features),
            "morph_logits": self.morph_head(features),
        }

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def head_parameters(self):
        for module in (self.species_head, self.sex_head, self.mating_head, self.morph_head):
            yield from module.parameters()

    def backbone_parameters(self):
        yield from self.backbone.parameters()


def build_model(
    backbone_name: str,
    num_species: int,
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> MultiTaskClassifier:
    return MultiTaskClassifier(
        backbone_name,
        ModelDimensions(species=num_species),
        pretrained=pretrained,
        dropout=dropout,
    )
