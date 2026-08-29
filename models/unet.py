import torch
from diffusers import UNet2DModel, DiTTransformer2DModel


# ---------------------------------------------------------------------------
# Architecture → recommended HuggingFace pretrained checkpoint
#
#  unet            : google/ddpm-celebahq-256
#                    Standard 2D DDPM U-Net, 256×256 RGB. Fast to fine-tune
#                    for grayscale MRI (conv_in/out are replaced).
#
#  attention_unet  : benetraco/brain_ddpm_256
#                    DDPM U-Net trained on brain MRI FLAIR slices, 256×256,
#                    1-channel, with full cross-attention blocks throughout.
#                    Best transfer learning starting point for MRI tasks.
#
#  dit             : facebook/DiT-XL-2-256
#                    Official DiT-XL/2 model (Peebles & Xie, 2023).
#                    256×256 class-conditional, FID=2.27 on ImageNet.
#                    Loaded as transformer weights; VAE / scheduler
#                    are configured separately during training.
# ---------------------------------------------------------------------------

ARCH_PRETRAINED_DEFAULTS: dict = {
    "unet":           "google/ddpm-celebahq-256",
    "attention_unet": "benetraco/brain_ddpm_256",
    "dit":            "facebook/DiT-XL-2-256",
}

# Scheduler configs to use for each architecture
# (used by train.py to load the matching DDPM scheduler)
ARCH_SCHEDULER_DEFAULTS: dict = {
    "unet":           "google/ddpm-celebahq-256",
    "attention_unet": "benetraco/brain_ddpm_256",
    "dit":            "google/ddpm-celebahq-256",   # DiT scheduler = standard DDPM
}

_OLD_GLOBAL_DEFAULT = "google/ddpm-celebahq-256"


def get_default_pretrained(model_type: str) -> str:
    """Return the recommended HuggingFace model ID for a given architecture."""
    if model_type not in ARCH_PRETRAINED_DEFAULTS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choices: {list(ARCH_PRETRAINED_DEFAULTS.keys())}"
        )
    return ARCH_PRETRAINED_DEFAULTS[model_type]


def get_default_scheduler_model(model_type: str) -> str:
    """Return the scheduler source model for a given architecture."""
    return ARCH_SCHEDULER_DEFAULTS.get(model_type, _OLD_GLOBAL_DEFAULT)


def get_model(
    model_type: str = "unet",
    pretrained_model_name=None,
    in_channels: int = 1,
    out_channels: int = 1,
    img_size: int = 256,
    attention_head_dim: int = 8,
):
    """
    Factory function to load and adapt different diffusion model architectures.

    Parameters
    ----------
    model_type            : "unet" | "attention_unet" | "dit"
    pretrained_model_name : HuggingFace repo ID.  If None, or if the legacy
                            global default is passed for a non-unet arch, the
                            architecture-specific default is used instead.
    in_channels           : Input channels (1 = MRI grayscale).
    out_channels          : Output channels.
    img_size              : Spatial size of images.
    attention_head_dim    : Attention head dimension.
    """
    # Auto-select the best pretrained model when caller passes None or the old
    # global default for an architecture that has a better-fitting checkpoint.
    if pretrained_model_name is None or (
        pretrained_model_name == _OLD_GLOBAL_DEFAULT and model_type != "unet"
    ):
        pretrained_model_name = get_default_pretrained(model_type)
        print(
            f"[get_model] Auto-selected pretrained checkpoint for '{model_type}': "
            f"{pretrained_model_name}"
        )
    else:
        print(
            f"[get_model] Using pretrained checkpoint for '{model_type}': "
            f"{pretrained_model_name}"
        )

    # -----------------------------------------------------------------------
    # 1. Plain U-Net  (default: google/ddpm-celebahq-256)
    #    Load pretrained weights, then replace conv_in/conv_out to match target
    #    channel count (typically 1 for MRI grayscale).
    # -----------------------------------------------------------------------
    if model_type == "unet":
        model = UNet2DModel.from_pretrained(
            pretrained_model_name,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )
        _adapt_channels(model, in_channels, out_channels)
        return model

    # -----------------------------------------------------------------------
    # 2. Attention U-Net  (default: benetraco/brain_ddpm_256)
    #    This checkpoint is already 1-channel and 256×256 with full attention,
    #    making it the ideal starting point for MRI fine-tuning.
    #    We still adapt channels in case the user requests a different count.
    # -----------------------------------------------------------------------
    elif model_type == "attention_unet":
        try:
            model = UNet2DModel.from_pretrained(
                pretrained_model_name,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
            _adapt_channels(model, in_channels, out_channels)
            print(
                f"[get_model] Loaded attention UNet from '{pretrained_model_name}'. "
                f"Adapted to in_channels={in_channels}, out_channels={out_channels}."
            )
        except Exception as e:
            print(
                f"[get_model] WARNING: Could not load '{pretrained_model_name}': {e}\n"
                "  Falling back to randomly-initialised attention UNet."
            )
            model = UNet2DModel(
                sample_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D", "DownBlock2D", "AttnDownBlock2D",
                    "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D",
                ),
                up_block_types=(
                    "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D",
                    "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
                ),
                attention_head_dim=attention_head_dim,
            )
        return model

    # -----------------------------------------------------------------------
    # 3. Diffusion Transformer  (default: facebook/DiT-XL-2-256)
    #    DiT operates on *latent* patches (VAE-compressed latent space).
    #    We extract only the transformer backbone from the DiTPipeline.
    #    The VAE + DDPM scheduler are wired up separately during training.
    # -----------------------------------------------------------------------
    elif model_type == "dit":
        try:
            from diffusers import DiTPipeline
            dit_pipeline = DiTPipeline.from_pretrained(pretrained_model_name)
            model = dit_pipeline.transformer   # DiTTransformer2DModel

            # Patch input projection if caller requests a non-standard channel count.
            # Standard DiT works in 4-channel VAE latent space.
            latent_channels = model.config.in_channels  # usually 4
            if in_channels != latent_channels:
                print(
                    f"[get_model] DiT expects {latent_channels}-ch latents; "
                    f"user requested {in_channels} — patching patch_embed projection."
                )
                old_proj = model.pos_embed.proj
                model.pos_embed.proj = torch.nn.Conv2d(
                    in_channels,
                    old_proj.out_channels,
                    kernel_size=old_proj.kernel_size,
                    stride=old_proj.stride,
                )
            print(
                f"[get_model] Loaded DiT transformer backbone from '{pretrained_model_name}'."
            )
        except Exception as e:
            print(
                f"[get_model] WARNING: Could not load DiT from '{pretrained_model_name}': {e}\n"
                "  Falling back to randomly-initialised DiTTransformer2DModel."
            )
            num_heads = max(1, img_size // (attention_head_dim * 4))
            model = DiTTransformer2DModel(
                sample_size=img_size // 8,
                in_channels=in_channels,
                out_channels=out_channels,
                num_layers=12,
                attention_head_dim=attention_head_dim,
                num_attention_heads=num_heads,
                patch_size=2,
                num_embeds_ada_norm=1000,
            )
        return model

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choices: {list(ARCH_PRETRAINED_DEFAULTS.keys())}"
        )


# ---------------------------------------------------------------------------
# Helper: adapt conv_in / conv_out channel counts in-place
# ---------------------------------------------------------------------------

def _adapt_channels(model: UNet2DModel, in_channels: int, out_channels: int):
    """Replace conv_in / conv_out so the model matches the target channel count."""
    block_ch = model.config.block_out_channels

    with torch.no_grad():
        if model.conv_in.in_channels != in_channels:
            model.conv_in = torch.nn.Conv2d(
                in_channels, block_ch[0], kernel_size=3, padding=1
            )
        if model.conv_out.out_channels != out_channels:
            model.conv_out = torch.nn.Conv2d(
                block_ch[0], out_channels, kernel_size=3, padding=1
            )
