import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
from torchmetrics.image.fid import FrechetInceptionDistance

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.m3_score_v2 import M3V2Metric

def load_images_from_dir(directory, num_samples=100):
    images = []
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files = sorted(files)[:num_samples]
    for f in files:
        img = Image.open(os.path.join(directory, f)).convert("RGB")
        images.append(img)
    return images

def main():
    real_dir = "data_mri/brats_axial_multislice"
    gen_base_dir = "medigan_samples"
    output_csv = "comparative_output/medigan_evaluation.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # M3V2Metric preprocesses internally — load raw [0,1] tensors
    m3_metric = M3V2Metric(device=device)
    fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)

    transform_v2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Get 100 real BraTS images as baseline
    print(f"Loading real BraTS images from {real_dir}...")
    real_pils = load_images_from_dir(real_dir, 100)
    real_tensors_m3 = torch.stack([transform_v2(img) for img in real_pils])  # raw [0,1]

    # Prune redundant layers once on a representative subset
    print("Pruning M3-V2 layers via CKA...")
    m3_metric.prune_layers_via_cka(real_tensors_m3[:20])
    
    # Update FID with real images
    print("Updating FID with real images...")
    for img in real_pils:
        t_fid = torch.from_numpy(np.array(img.resize((299,299)))).permute(2,0,1).unsqueeze(0).to(device)
        fid_metric.update(t_fid, real=True)
        
    results = []

    # Iterate through medigan generated directories
    gen_dirs = [d for d in os.listdir(gen_base_dir) if os.path.isdir(os.path.join(gen_base_dir, d))]
    
    for gen_dir_name in gen_dirs:
        full_path = os.path.join(gen_base_dir, gen_dir_name)
        print(f"\nEvaluating: {gen_dir_name}")
        
        gen_pils = load_images_from_dir(full_path, 100)
        if not gen_pils:
            print(f"No images found in {gen_dir_name}, skipping.")
            continue
            
        gen_tensors_m3 = torch.stack([transform_v2(img) for img in gen_pils])  # raw [0,1]

        with torch.no_grad():
            m3_result = m3_metric(real_tensors_m3, gen_tensors_m3)

        total_m3  = m3_result["m3_v2_final_score"]
        layer_dists = m3_result.get("layer_distances", {})

        # FID
        fid_metric.reset()
        for img in gen_pils:
            t_fid = torch.from_numpy(np.array(img.resize((299, 299)))).permute(2, 0, 1).unsqueeze(0).to(device)
            fid_metric.update(t_fid, real=False)
        fid_score = fid_metric.compute().item()

        print(f"  FID: {fid_score:.4f}")
        print(f"  M3-V2 Score: {total_m3:.4f}")

        row = {
            "Model":    gen_dir_name,
            "FID":      fid_score,
            "M3-Score": total_m3,
        }
        row.update(layer_dists)   # add per-layer MMD² columns
        results.append(row)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nFull results saved to {output_csv}")
    print("\nSummary Table:")
    print(df[['Model', 'FID', 'M3-Score']].to_string(index=False))

if __name__ == "__main__":
    main()
