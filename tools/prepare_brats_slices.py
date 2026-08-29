import os
import argparse
import glob
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

def prepare_slices(data_dir, output_dir, modality, axes, threshold, depth=None, save_masks=False):
    if isinstance(axes, int):
        axes = [axes]
    print(f"Searching for BraTS cases in: {data_dir}")
    print(f"Target Modality: {modality.upper()} | Axes: {axes} | Output Dir: {output_dir}")
    if depth:
        if len(depth) == 1:
            if str(depth[0]).lower() == 'middle':
                print(f"Target Depth: Middle slice")
            else:
                print(f"Target Depth: Slice {depth[0]}")
        elif len(depth) >= 3 and str(depth[0]).lower() == 'middle_spaced':
            print(f"Target Depth: Middle {depth[1]} slices with spacing {depth[2]}")
        else:
            print(f"Target Depth Range: Slices {depth[0]} to {depth[1]}")
    else:
        print("Target Depth: All valid slices")
    
    # We look for files matching the modality inside patient subdirectories.
    search_pattern = os.path.join(data_dir, "*", f"*_{modality}.nii.gz")
    files = glob.glob(search_pattern)
    
    # If the directory structure is different (e.g. all files in one folder), fallback search
    if not files:
        search_pattern = os.path.join(data_dir, f"*_{modality}.nii.gz")
        files = glob.glob(search_pattern)
        
    if not files:
        print(f"Error: No *_{modality}.nii.gz files found in {data_dir} or its immediate subdirectories.")
        return

    print(f"Found {len(files)} {modality.upper()} volumes. Processing...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    
    for file_path in tqdm(files, desc="Extracting Slices"):
        # e.g., "BraTS2021_00000"
        case_id = os.path.basename(os.path.dirname(file_path))
        if case_id == os.path.basename(data_dir):
            # if files are in the main dir, use filename prefix
            case_id = os.path.basename(file_path).split('_')[0] + "_" + os.path.basename(file_path).split('_')[1]

        # Locate the co-registered segmentation volume (*_seg.nii.gz) when requested.
        # BraTS convention: file sits in the same directory as the modality file.
        seg_data = None
        if save_masks:
            seg_path = os.path.join(os.path.dirname(file_path), f"{case_id}_seg.nii.gz")
            if not os.path.isfile(seg_path):
                # Try flat-directory layout: replace modality suffix with _seg
                seg_path = file_path.replace(f"_{modality}.nii.gz", "_seg.nii.gz")
            if os.path.isfile(seg_path):
                seg_img  = nib.load(seg_path)
                seg_data = (seg_img.get_fdata() > 0).astype(np.uint8) * 255
            else:
                print(f"  [mask] seg not found for {case_id}, skipping mask extraction")

        try:
            # 1. Load NIfTI 3D Array
            img = nib.load(file_path)
            data = img.get_fdata() # Shape usually [W, H, D]
            
            # 2. Robust Normalization (Ignore background 0s for percentiles)
            brain_mask = data > 0
            if not np.any(brain_mask):
                continue # Skip completely empty volumes
                
            p1 = np.percentile(data[brain_mask], 1)
            p99 = np.percentile(data[brain_mask], 99)
            
            # Clip outlier high-intensities, then normalize [0, 1]
            data = np.clip(data, p1, p99)
            if p99 > p1:
                data = (data - p1) / (p99 - p1)
            else:
                data = np.zeros_like(data)
                
            # Scale to 8-bit [0, 255]
            data = (data * 255.0).astype(np.uint8)
            
            # 3. Axis Extraction
            for axis in axes:
                if axis == 0:    # Sagittal
                    num_slices = data.shape[0]
                elif axis == 1:  # Coronal
                    num_slices = data.shape[1]
                elif axis == 2:  # Axial (Standard top-down view)
                    num_slices = data.shape[2]
                else:
                    raise ValueError("Axis must be 0, 1, or 2.")
                    
                # --- NEW: Depth Range Logic ---
                indices_to_extract = []
                
                if depth is not None:
                    if len(depth) == 1:
                        if str(depth[0]).lower() == 'middle':
                            indices_to_extract = [num_slices // 2]
                        else:
                            indices_to_extract = [int(depth[0])]
                    elif len(depth) >= 3 and str(depth[0]).lower() == 'middle_spaced':
                        # Usage: --depth middle_spaced <num_slices> <spacing>
                        n_slices = int(depth[1])
                        spacing = int(depth[2])
                        mid = num_slices // 2
                        start = mid - (n_slices // 2) * spacing
                        
                        for step in range(n_slices):
                            indices_to_extract.append(start + step * spacing)
                    elif len(depth) >= 2:
                        start_idx = min(int(depth[0]), int(depth[1]))
                        end_idx = max(int(depth[0]), int(depth[1]))
                        indices_to_extract = list(range(start_idx, end_idx + 1))
                else:
                    indices_to_extract = list(range(num_slices))
                
                # Ensure indices don't go out of bounds for the current volume
                valid_indices = []
                for idx in indices_to_extract:
                    if 0 <= idx < num_slices:
                        valid_indices.append(idx)
                
                # If no valid indices, skip
                if not valid_indices:
                    continue
                # ------------------------------

                # 4. Iterate and Save 2D Slices
                for i in valid_indices:
                    if axis == 0:
                        slice_data = data[i, :, :]
                    elif axis == 1:
                        slice_data = data[:, i, :]
                    elif axis == 2:
                        slice_data = data[:, :, i]
                        
                    # 5. Filter out empty slices (must have more than `threshold`% non-zero pixels)
                    non_zero_ratio = np.count_nonzero(slice_data) / slice_data.size
                    if non_zero_ratio < threshold:
                        continue
                        
                    # 6. Correct Orientation (NIfTI arrays often need a 90 degree rotation for PNG viewing)
                    if axis == 2:
                        slice_data = np.rot90(slice_data)
                    
                    # Convert to Python Image Library grayscale image
                    img_pil = Image.fromarray(slice_data, mode='L')

                    # Save slice as PNG
                    if len(axes) > 1:
                        axis_name = {0: 'sagittal', 1: 'coronal', 2: 'axial'}[axis]
                        filename = f"{case_id}_{axis_name}_slice{i:03d}.png"
                    else:
                        filename = f"{case_id}_slice{i:03d}.png"
                    save_path = os.path.join(output_dir, filename)
                    img_pil.save(save_path)
                    saved_count += 1

                    # Co-extract binary segmentation mask at the same slice / axis.
                    # Any label > 0 is mapped to 255 (tumor present); background = 0.
                    if save_masks and seg_data is not None:
                        if axis == 0:
                            seg_slice = seg_data[i, :, :]
                        elif axis == 1:
                            seg_slice = seg_data[:, i, :]
                        else:
                            seg_slice = seg_data[:, :, i]
                        if axis == 2:
                            seg_slice = np.rot90(seg_slice)
                        mask_filename = f"{case_id}_segmask_slice{i:03d}.png"
                        mask_pil = Image.fromarray(seg_slice, mode='L')
                        mask_pil.save(os.path.join(output_dir, mask_filename))
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    print(f"\nExtraction Complete!")
    print(f"Successfully saved {saved_count} structured 2D slices to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and normalize 2D slices from 3D BraTS NIfTI files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the main BraTS dataset folder containing patient subfolders.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the extracted 2D PNG slices.")
    parser.add_argument("--modality", type=str, default="flair", choices=["flair", "t1", "t1ce", "t2"], help="Which modality to extract (default: flair).")
    parser.add_argument("--axis", type=int, nargs='+', default=[2], choices=[0, 1, 2], help="Slicing axis: 0 (Sagittal), 1 (Coronal), 2 (Axial). Can specify multiple. Default is 2 (Axial).")
    parser.add_argument("--threshold", type=float, default=0.0001, help="Minimum ratio of non-zero (brain) pixels required to save a slice. Filters empty edges. Default 0.05 (5%).")
    
    # --- Depth Argument ---
    parser.add_argument("--depth", type=str, nargs='+', help="Optional. Specify a single index (--depth 75), a range (--depth 50 100), 'middle', or 'middle_spaced N S' to get N slices around the middle separated by S.")
    parser.add_argument("--save_masks", action="store_true", help="Also extract binary tumor segmentation masks from *_seg.nii.gz (any label > 0 = 255). Saved as {case_id}_segmask_slice{i:03d}.png alongside the modality slices.")

    args = parser.parse_args()
    prepare_slices(args.data_dir, args.output_dir, args.modality, args.axis, args.threshold, args.depth, args.save_masks)