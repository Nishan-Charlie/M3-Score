
import torch
import numpy as np
from evaluation.m3_score_v2 import M3V2Metric
from PIL import Image
import glob
import os

def check_cka():
    metric = M3V2Metric(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load 10 images
    paths = glob.glob("data_mri/brats_axial_multislice/*.png")[:10]
    imgs = [Image.open(p).convert("RGB") for p in paths]
    
    # Preprocess
    img_tensor = torch.stack([metric.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0) for img in imgs]).to(metric.device)
    
    # Extract features
    feats, _ = metric._extract_features_and_entropy(img_tensor)
    
    # Compute CKA matrix
    n_layers = len(feats)
    cka_matrix = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            cka_matrix[i, j] = metric._linear_cka(feats[i], feats[j])
            
    print("CKA Matrix (Top-left 4x4):")
    print(cka_matrix[:4, :4])
    print("\nMean off-diagonal CKA:", (cka_matrix.sum() - n_layers) / (n_layers * (n_layers - 1)))
    
    # Print max CKA between adjacent layers
    adj_cka = [cka_matrix[i, i+1] for i in range(n_layers-1)]
    print("\nAdjacent layers CKA:", adj_cka)
    print("Min adjacent:", min(adj_cka))
    print("Max adjacent:", max(adj_cka))

if __name__ == "__main__":
    check_cka()
