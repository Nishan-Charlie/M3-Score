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


@pytest.fixture()
def offline_backbone(monkeypatch, tiny_model):
    """Resolve the default (m3-v1) backbone to the tiny random ViT, so nothing is downloaded."""
    from m3score import backbone, metric

    def load(spec=None, device="cpu"):
        if spec is None:
            return Backbone(tiny_model.to(device), "tiny-test-vit")
        return backbone.load_backbone(spec, device)

    monkeypatch.setattr(metric, "load_backbone", load)
