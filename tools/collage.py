import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import random, os, glob
random.seed(42)

real_dir = "data_mri/brats_axial_multislice"
gen_dir  = "output/generated_500_best"
out_dir  = "results/collages"
os.makedirs(out_dir, exist_ok=True)

def make_collage(img_dir, out_path, title, n=36, cols=6, img_size=128):
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) +
                   glob.glob(os.path.join(img_dir, "*.jpg")))
    random.shuffle(paths)
    paths = paths[:n]
    rows = n // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6), dpi=150)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.04, wspace=0.04)

    for ax, p in zip(axes.flat, paths):
        img = Image.open(p).convert("L").resize((img_size, img_size), Image.LANCZOS)
        ax.imshow(np.array(img), cmap="gray", vmin=0, vmax=255)
        ax.axis("off")

    # Hide any unused axes
    for ax in axes.flat[len(paths):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", color="#222222", y=1.01)
    plt.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

make_collage(real_dir, f"{out_dir}/collage_real_36.png", "BraTS Real MRI Samples Images")
make_collage(gen_dir,  f"{out_dir}/collage_generated_36.png", "DDPM Generated MRI Samples Images")
