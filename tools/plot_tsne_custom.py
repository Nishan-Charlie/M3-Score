import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import models, transforms
from transformers import AutoModel, AutoImageProcessor
from tqdm.auto import tqdm
import argparse

# ---------------------------------------------------------------------------
# Feature Extractors
# ---------------------------------------------------------------------------

class InceptionV3Extractor(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        # Inception V3 expects 299x299 input by default, or we can resize.
        self.backbone = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        self.backbone.fc = nn.Identity()  # Remove final linear layer to get 2048-dim features
        self.backbone.eval()
        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        # InceptionV3 returns a tuple (logits, aux_logits) during training, but only logits in eval
        out = self.backbone(x)
        return out  # (B, 2048)


class RadioDinoExtractor(nn.Module):
    def __init__(self, checkpoint_path=None, device="cpu"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
        self.model = AutoModel.from_pretrained("microsoft/rad-dino")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading RadioDino checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            print("No RadioDino checkpoint found, using base microsoft/rad-dino.")
            
        self.model.eval()
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def forward(self, images: list):
        # We process PIL images directly for RadioDino as it comes with its own processor
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Use CLS token, shape: (B, 768)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_paths(directory: str, max_n=None) -> list:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    paths = sorted(paths)
    if max_n:
        paths = paths[:max_n]
    return paths

_INCEPTION_TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_inception(paths, extractor, batch_size=32):
    all_feats = []
    for start in tqdm(range(0, len(paths), batch_size), desc="InceptionV3 Ext"):
        batch_paths = paths[start : start + batch_size]
        imgs = [_INCEPTION_TRANSFORM(Image.open(p).convert("RGB")) for p in batch_paths]
        feats = extractor(torch.stack(imgs))
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)

def extract_radiodino(paths, extractor, batch_size=32):
    all_feats = []
    for start in tqdm(range(0, len(paths), batch_size), desc="RadioDino Ext"):
        batch_paths = paths[start : start + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        feats = extractor(imgs)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def produce_plot(embedding, labels, title, out_path, n_real, n_gen, perplexity, plot_groups=None):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    palette = {
        "Real": ("#4fc3f7", "o", 45),   # sky blue circles
        "Generated": ("#ef9a9a", "^", 40),   # rose triangles
    }

    if plot_groups is None:
        plot_groups = ["Real", "Generated"]

    # Precompute global limits to keep axes stable across sub-plots
    g_xmin, g_xmax = embedding[:, 0].min() - 5, embedding[:, 0].max() + 5
    g_ymin, g_ymax = embedding[:, 1].min() - 5, embedding[:, 1].max() + 5

    for grp, (color, marker, size) in palette.items():
        if grp not in plot_groups:
            continue
        mask = labels == grp
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, marker=marker, s=size, alpha=0.70, linewidths=0, label=grp,
        )

    # Optional contours
    try:
        from scipy.stats import gaussian_kde
        for grp, (color, _, _) in palette.items():
            if grp not in plot_groups:
                continue
            mask = labels == grp
            pts = embedding[mask].T
            if pts.shape[1] > 2:
                kde = gaussian_kde(pts)
                xx, yy = np.mgrid[g_xmin:g_xmax:100j, g_ymin:g_ymax:100j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=5, colors=color, alpha=0.3, linewidths=0.8)
    except Exception:
        pass

    ax.set_xlim([g_xmin, g_xmax])
    ax.set_ylim([g_ymin, g_ymax])

    legend = ax.legend(framealpha=0.15, facecolor="#1a1a1a", edgecolor="#444", 
                       fontsize=12, markerscale=1.5, labelcolor="white")

    ax.set_title(title, color="white", fontsize=15, pad=14)
    ax.set_xlabel("t-SNE dim 1", color="#aaa", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", color="#aaa", fontsize=11)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    ax.text(
        0.01, 0.99,
        f"Real n={n_real} | Gen n={n_gen}\nperplexity={perplexity}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=9, color="#aaa", path_effects=[pe.withStroke(linewidth=2, foreground="black")],
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--gen_dir", default=None)
    parser.add_argument("--output_dir", default="./tsne_custom_output")
    parser.add_argument("--rad_ckpt", default="output/radiodino_segmentation/backbone_final.pth")
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    real_paths = _load_paths(args.real_dir, args.num_images)
    gen_paths = _load_paths(args.gen_dir, args.num_images) if args.gen_dir else []
    n_real, n_gen = len(real_paths), len(gen_paths)
    
    labels_list = ["Real"] * n_real
    if n_gen > 0:
        labels_list += ["Generated"] * n_gen
    labels = np.array(labels_list)

    print(f"Loaded {n_real} real, {n_gen} gen images.")

    # 1. Inception V3
    print("\n--- Running Inception V3 ---")
    inc_ext = InceptionV3Extractor(args.device)
    real_inc = extract_inception(real_paths, inc_ext)
    if n_gen > 0:
        gen_inc = extract_inception(gen_paths, inc_ext)
        all_inc = np.concatenate([real_inc, gen_inc], axis=0)
    else:
        all_inc = real_inc

    print("Computing t-SNE for InceptionV3...")
    tsne_inc = TSNE(n_components=2, perplexity=args.perplexity, max_iter=args.n_iter, random_state=42, init="pca", learning_rate="auto")
    emb_inc = tsne_inc.fit_transform(all_inc)
    
    if n_gen > 0:
        produce_plot(emb_inc, labels, "t-SNE: InceptionV3 Features", os.path.join(args.output_dir, "tsne_inceptionV3_combined.png"), n_real, n_gen, args.perplexity)
        produce_plot(emb_inc, labels, "t-SNE: InceptionV3 (Real Only)", os.path.join(args.output_dir, "tsne_inceptionV3_real.png"), n_real, n_gen, args.perplexity, plot_groups=["Real"])
        produce_plot(emb_inc, labels, "t-SNE: InceptionV3 (Generated Only)", os.path.join(args.output_dir, "tsne_inceptionV3_gen.png"), n_real, n_gen, args.perplexity, plot_groups=["Generated"])
    else:
        produce_plot(emb_inc, labels, "t-SNE: InceptionV3 (Real Only)", os.path.join(args.output_dir, "tsne_inceptionV3_real_only.png"), n_real, n_gen, args.perplexity, plot_groups=["Real"])
        
    del inc_ext, real_inc, all_inc
    if n_gen > 0:
        del gen_inc
    torch.cuda.empty_cache()

    # 2. RadioDino
    print("\n--- Running RadioDino ---")
    rad_ext = RadioDinoExtractor(args.rad_ckpt, args.device)
    real_rad = extract_radiodino(real_paths, rad_ext)
    if n_gen > 0:
        gen_rad = extract_radiodino(gen_paths, rad_ext)
        all_rad = np.concatenate([real_rad, gen_rad], axis=0)
    else:
        all_rad = real_rad

    print("Computing t-SNE for RadioDino...")
    tsne_rad = TSNE(n_components=2, perplexity=args.perplexity, max_iter=args.n_iter, random_state=42, init="pca", learning_rate="auto")
    emb_rad = tsne_rad.fit_transform(all_rad)

    if n_gen > 0:
        produce_plot(emb_rad, labels, "t-SNE: RadioDino Features", os.path.join(args.output_dir, "tsne_RadioDino_combined.png"), n_real, n_gen, args.perplexity)
        produce_plot(emb_rad, labels, "t-SNE: RadioDino (Real Only)", os.path.join(args.output_dir, "tsne_RadioDino_real.png"), n_real, n_gen, args.perplexity, plot_groups=["Real"])
        produce_plot(emb_rad, labels, "t-SNE: RadioDino (Generated Only)", os.path.join(args.output_dir, "tsne_RadioDino_gen.png"), n_real, n_gen, args.perplexity, plot_groups=["Generated"])
    else:
        produce_plot(emb_rad, labels, "t-SNE: RadioDino (Real Only)", os.path.join(args.output_dir, "tsne_RadioDino_real_only.png"), n_real, n_gen, args.perplexity, plot_groups=["Real"])

if __name__ == "__main__":
    main()
