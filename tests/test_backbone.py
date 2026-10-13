import numpy as np
import pytest
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from _legacy import load_legacy_metric, load_legacy_module
from m3score import M3ScoreError, protocol
from m3score.backbone import Backbone, load_backbone

LAYERS = (12, 9, 4)


def test_extract_returns_cls_features_for_the_requested_blocks(tiny_backbone):
    prepared = tiny_backbone.prepare(torch.rand(5, 3, 32, 32))
    features = tiny_backbone.extract(prepared, LAYERS, "cpu")
    assert set(features) == {12, 9, 4}
    assert all(f.shape == (5, 48) and f.dtype == torch.float32 for f in features.values())
    assert not torch.equal(features[4], features[12])


def test_prepare_repeats_grey_and_resizes(tiny_backbone):
    prepared = tiny_backbone.prepare(torch.rand(2, 1, 50, 40))
    assert prepared.shape == (2, 3, 32, 32)
    assert torch.equal(prepared[:, 0], prepared[:, 1])
    assert float(prepared.min()) >= 0.0 and float(prepared.max()) <= 1.0


def test_prepare_leaves_native_size_untouched(tiny_backbone):
    images = torch.rand(2, 3, 32, 32)
    assert torch.equal(tiny_backbone.prepare(images), images)


def test_features_do_not_depend_on_how_images_are_batched(tiny_backbone):
    prepared = tiny_backbone.prepare(torch.rand(6, 3, 32, 32))
    whole = tiny_backbone.extract(prepared, LAYERS, "cpu")
    halves = [tiny_backbone.extract(prepared[i : i + 3], LAYERS, "cpu") for i in (0, 3)]
    for layer in LAYERS:
        assert torch.allclose(whole[layer], torch.cat([h[layer] for h in halves]), atol=1e-5)


def test_extraction_matches_research_code(tiny_model, tiny_backbone):
    module = load_legacy_module()
    legacy = load_legacy_metric()
    legacy.backbone = tiny_model
    legacy._backend = "timm_vit"
    legacy.num_layers = 12
    legacy._norm_mean = module._IMAGENET_MEAN
    legacy._norm_std = module._IMAGENET_STD
    images = torch.rand(20, 3, 32, 32)
    research = legacy._extract_cls_features(images, {4, 9, 12})
    package = tiny_backbone.extract(tiny_backbone.prepare(images), LAYERS, "cpu")
    for layer in LAYERS:
        assert torch.allclose(package[layer], research[layer - 1], atol=1e-5)


def test_rejects_a_model_that_is_not_a_vision_transformer():
    with pytest.raises(TypeError, match="vision transformer"):
        Backbone(nn.Linear(2, 2), "linear")


def test_load_backbone_accepts_a_module(tiny_model):
    backbone = load_backbone(tiny_model, "cpu")
    assert backbone.name == "VisionTransformer"
    assert backbone.num_layers == 12
    assert backbone.image_size == (32, 32)


def test_load_backbone_rejects_other_types():
    with pytest.raises(TypeError, match="backbone"):
        load_backbone(123, "cpu")


def test_default_backbone_is_the_pinned_revision(monkeypatch, tiny_model):
    calls = {}

    def fake_create(name, **kwargs):
        calls["name"], calls["kwargs"] = name, kwargs
        return tiny_model

    monkeypatch.setattr(timm, "create_model", fake_create)
    backbone = load_backbone(None, "cpu")
    pinned = f"{protocol.BACKBONE_ID}@{protocol.BACKBONE_REVISION}"
    assert calls["name"] == f"hf_hub:{pinned}"
    assert calls["kwargs"] == {"pretrained": True, "img_size": 224}
    assert backbone.name == pinned


def test_backbone_download_failure_explains_itself(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(timm, "create_model", refuse)
    with pytest.raises(M3ScoreError, match="Hugging Face") as info:
        load_backbone(None, "cpu")
    assert "HF_HUB_OFFLINE" in str(info.value)
    assert "connection refused" in str(info.value)


def test_prepare_is_a_pillow_bilinear_resize_in_float(tiny_backbone):
    # the resize is a frozen part of the protocol: same filter as Pillow, no 8-bit rounding
    image = torch.rand(1, 1, 100, 80, generator=torch.Generator().manual_seed(0))
    pillow = Image.fromarray(image[0, 0].numpy()).resize((32, 32), Image.BILINEAR)
    prepared = tiny_backbone.prepare(image)
    assert np.allclose(prepared[0, 0].numpy(), np.asarray(pillow, dtype=np.float32), atol=1e-4)


def test_prepare_antialiases_when_shrinking(tiny_backbone):
    image = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(0))
    aliased = F.interpolate(
        image, size=(32, 32), mode="bilinear", antialias=False, align_corners=False
    )
    assert float((tiny_backbone.prepare(image) - aliased).abs().max()) > 0.05
