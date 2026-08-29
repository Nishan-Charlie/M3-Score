import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

def extract_masks(data_dir, output_dir, target_slice=77):
    print(f"Searching for BraTS segmentation masks in: {data_dir}")
    print(f"Output Dir: {output_dir}")
    
    search_pattern = os.path.join(data_dir, "*", "*_seg.nii.gz")
    files = glob.glob(search_pattern)
        
    if not files:
        print(f"Error: No *_seg.nii.gz files found in {data_dir} or its immediate subdirectories.")
        return

    print(f"Found {len(files)} SEG volumes. Processing...")
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    unique_labels = set()
    
    for file_path in tqdm(files, desc="Extracting Masks"):
        case_id = os.path.basename(os.path.dirname(file_path))
            
        try:
            # Load NIfTI 3D Array
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # Ensure index is within bounds
            num_slices = data.shape[2]
            if target_slice >= num_slices or target_slice < 0:
                print(f"Skipping {case_id}: Slice {target_slice} is out of bounds (max {num_slices-1}).")
                continue
                
            # Extract axial slice at target_index
            slice_data = data[:, :, target_slice]
            
            # Collect unique labels just to verify mask integers
            for val in np.unique(slice_data):
                unique_labels.add(int(val))
                
            # NIfTI arrays often need a 90 degree rotation for PNG viewing
            slice_data = np.rot90(slice_data)
            
            # Convert to uint8 without normalization (keeping 0, 1, 2, 4)
            slice_data = slice_data.astype(np.uint8)
            img_pil = Image.fromarray(slice_data, mode='L')
            
            # Save format matching standard slices: e.g. BraTS2021_00000_slice077.png
            filename = f"{case_id}_slice{target_slice:03d}.png"
            save_path = os.path.join(output_dir, filename)
            img_pil.save(save_path)
            saved_count += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    print(f"\nExtraction Complete!")
    print(f"Successfully saved {saved_count} 2D mask slices to: {output_dir}")
    print(f"Unique pixel values encountered across all masks: {sorted(list(unique_labels))}")

if __name__ == "__main__":
    DATA_DIR = "/home/e21283/mediGAN/mri-diffuser/huggingface_models/data_mri/brats"
    OUTPUT_DIR = "/home/e21283/mediGAN/mri-diffuser/huggingface_models/data_mri/brats_axial_masks"
    TARGET_SLICE = 77
    extract_masks(DATA_DIR, OUTPUT_DIR, TARGET_SLICE)
