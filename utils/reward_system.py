import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import torch.nn.functional as F


class BaseReward(nn.Module, ABC):
    """Abstract base class for modular DDPO rewards."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculate the reward for a batch of images.

        Args:
            images (torch.Tensor): Generated images, shape (B, C, H, W).
            **kwargs: Additional tensors needed for reward calculation.

        Returns:
            torch.Tensor: 1D tensor of rewards, shape (B,).
        """
        pass


class DeepCosineDiversityReward(BaseReward):
    """
    Diversity reward for unconditional generation.
    Computes negative cosine similarity between deep feature embeddings.
    Higher diversity among generated samples → higher reward.

    Args:
        feature_extractor: A frozen nn.Module (e.g. VGG-16 backbone) that
                           maps (B, C, H, W) → (B, D) feature vectors.
        scale:             Reward scaling factor.
    """

    def __init__(self, feature_extractor: nn.Module, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        B = images.shape[0]
        if B < 2:
            return torch.zeros(B, device=images.device)

        # Grayscale → RGB if the extractor expects 3 channels
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        features = self.feature_extractor(images)
        if features.dim() > 2:
            features = features.view(B, -1)

        features_norm = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.mm(features_norm, features_norm.t())

        # Mask out self-similarity on the diagonal
        mask = torch.eye(B, device=images.device, dtype=torch.bool)
        sim_matrix.masked_fill_(mask, 0.0)

        mean_sim = sim_matrix.sum(dim=1) / (B - 1)   # (B,)
        return self.scale * (-mean_sim)               # higher diversity = higher reward


class TrajectoryEfficiencyReward(BaseReward):
    """
    Pixel-level diversity reward — requires no external model.
    Good choice for a quick first test or when no GPU classifier is available.

    Reward = scale * pixel_variance - step_penalty * num_steps

    Encourages the model to generate visually diverse images in fewer
    denoising timesteps.

    Args:
        scale:        Weight on the diversity component.
        step_penalty: Penalty per denoising step (pass num_steps in forward kwargs).
    """

    def __init__(self, scale: float = 1.0, step_penalty: float = 0.05):
        super().__init__()
        self.scale = scale
        self.step_penalty = step_penalty

    def forward(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        num_steps = kwargs.get('num_steps', None)
        if num_steps is None:
            num_steps = torch.zeros(images.shape[0], device=images.device)

        B = images.shape[0]
        if B < 2:
            num_steps_t = torch.as_tensor(num_steps, dtype=torch.float32, device=images.device)
            if num_steps_t.dim() == 0:
                num_steps_t = num_steps_t.unsqueeze(0)
            return -self.step_penalty * num_steps_t

        shifted = torch.roll(images, shifts=1, dims=0)
        mse = torch.mean((images - shifted) ** 2, dim=[1, 2, 3])   # (B,)

        if not isinstance(num_steps, torch.Tensor):
            num_steps_t = torch.full((B,), float(num_steps), device=images.device)
        else:
            num_steps_t = num_steps.to(dtype=torch.float32, device=images.device)
            if num_steps_t.numel() == 1:
                num_steps_t = num_steps_t.expand(B)

        return self.scale * mse - self.step_penalty * num_steps_t


class RewardFactory:
    """
    Factory to instantiate rewards by name.

    Available rewards
    -----------------
    'trajectory_efficiency'  — No external model needed.
                               Rewards pixel-level diversity and fast generation.
                               Best starting point.

    'deep_cosine_diversity'  — Requires a feature_extractor (e.g. VGG-16 backbone).
                               Rewards feature-level diversity between generated samples.
                               Better at capturing semantic / perceptual diversity.

    Example
    -------
    # Trajectory efficiency (no external model required)
    reward = RewardFactory.create('trajectory_efficiency', scale=1.0)

    # Deep cosine diversity (pass a VGG-16 backbone)
    import torchvision.models as m
    vgg = m.vgg16(weights=m.VGG16_Weights.DEFAULT)
    vgg.classifier = torch.nn.Identity()
    reward = RewardFactory.create('deep_cosine_diversity', feature_extractor=vgg, scale=1.0)
    """

    AVAILABLE = ["trajectory_efficiency", "deep_cosine_diversity"]

    @staticmethod
    def create(reward_type: str, **kwargs) -> BaseReward:
        if reward_type == "trajectory_efficiency":
            return TrajectoryEfficiencyReward(
                scale=kwargs.get("scale", 1.0),
                step_penalty=kwargs.get("step_penalty", 0.05),
            )
        elif reward_type == "deep_cosine_diversity":
            if "feature_extractor" not in kwargs:
                raise ValueError(
                    "DeepCosineDiversityReward requires a 'feature_extractor' kwarg "
                    "(e.g. a torchvision VGG-16 backbone with the classifier head removed)."
                )
            return DeepCosineDiversityReward(
                feature_extractor=kwargs["feature_extractor"],
                scale=kwargs.get("scale", 1.0),
            )
        else:
            raise ValueError(
                f"Unknown reward type: '{reward_type}'. "
                f"Choose from: {RewardFactory.AVAILABLE}"
            )


def build_reward_fn(reward_type: str, scale: float, device: str) -> BaseReward:
    """
    Convenience function: instantiate a reward by name and move it to device.

    Parameters
    ----------
    reward_type : "trajectory_efficiency" | "deep_cosine_diversity"
    scale       : Reward scaling factor.
    device      : torch device string (e.g. "cuda:0", "cpu").

    Returns None if reward_type is "none".
    """
    if reward_type == "none":
        return None

    if reward_type == "trajectory_efficiency":
        return RewardFactory.create("trajectory_efficiency", scale=scale).to(device)

    if reward_type == "deep_cosine_diversity":
        import torchvision.models as tv_models
        import torch
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        vgg.classifier = torch.nn.Identity()
        vgg = vgg.eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        return RewardFactory.create(
            "deep_cosine_diversity",
            feature_extractor=vgg,
            scale=scale,
        ).to(device)

    raise ValueError(
        f"Unknown reward_type '{reward_type}'. "
        "Choose from: none, trajectory_efficiency, deep_cosine_diversity"
    )
