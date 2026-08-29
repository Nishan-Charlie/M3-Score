import os
import sys
import torch
import numpy as np

# Add project root to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.m3_score_v2 import M3V2Metric
from experiments._shared_utils import get_m3_transform, load_pils_recursive

def run_m3_v2_benchmark(real_dir, gen_dir, n=100, device="cuda"):
    """
    Demonstrates the M3-V2 Metric on two image directories.
    """
    transform = get_m3_transform()
    metric = M3V2Metric(device=device)
    
    print(f"--- Loading data (n={n}) ---")
    real_pils = load_pils_recursive(real_dir, n=n)
    gen_pils = load_pils_recursive(gen_dir, n=n)
    
    real_imgs = torch.stack([transform(p.convert("RGB")) for p in real_pils])
    gen_imgs = torch.stack([transform(p.convert("RGB")) for p in gen_pils])
    
    # 1. Step: Prune layers using CKA on real images
    print("\n--- Phase 1: CKA Layer Pruning ---")
    metric.prune_layers_via_cka(real_imgs)
    metric.visualize_cka_matrix(real_imgs, "results/cka_matrix_radiodino.png")
    
    # 2. Step: Compute M3-V2 Score with Attention Aggregation and KID Consistency
    print("\n--- Phase 2: M3-V2 Score Computation ---")
    results = metric(real_imgs, gen_imgs)
    
    print("\nResults:")
    print(f"Final M3-V2 Score: {results['m3_v2_final_score']:.6f}")
    print("\nLayer-wise Details:")
    for layer in results['active_layers']:
        l_id = f"L{layer}"
        w = results['layer_weights'][l_id]
        d = results['layer_distances'][l_id]
        s = results['layer_stabilities'][l_id]
        print(f"  {l_id}: Distance={d:.4f}, Stability={s:.4f}, Weight={w:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    run_m3_v2_benchmark(args.real_dir, args.gen_dir, n=args.n, device=args.device)
