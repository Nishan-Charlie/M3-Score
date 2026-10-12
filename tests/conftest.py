import pytest
import timm
import torch

from m3score.backbone import Backbone


@pytest.fixture(scope="session")
def tiny_model():
    """A randomly initialised 12-block ViT for 32x32 inputs: fast, and needs no download."""
    torch.manual_seed(0)
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=0,
        img_size=32,
        patch_size=8,
        embed_dim=48,
        depth=12,
        num_heads=3,
    )
    return model.eval()


@pytest.fixture()
def tiny_backbone(tiny_model):
    return Backbone(tiny_model, "tiny-test-vit")
