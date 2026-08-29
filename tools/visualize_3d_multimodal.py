import argparse
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.colors import ListedColormap
import nibabel as nib

def load_nifti(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        if len(data.shape) == 4:
            data = data[..., 0]
        return data
    except Exception as e:
        print(f"Failed to load {file_path}. Error: {e}")
        return None

def normalize(data):
    mask = data > 0
    if not np.any(mask):
        return data
    p1 = np.percentile(data[mask], 1)
    p99 = np.percentile(data[mask], 99)
    data = np.clip(data, p1, p99)
    if p99 > p1:
        data = (data - p1) / (p99 - p1)
    return data

def main():
    parser = argparse.ArgumentParser(description="Visualize and interact with 3D multi-modal BraTS data.")
    parser.add_argument("folder_path", type=str, nargs='?', 
                        default="datasets/brats/BraTS2021_00186",
                        help="Path to the patient folder containing .nii.gz files")
    
    args = parser.parse_args()
    folder_path = args.folder_path
    
    if not os.path.isdir(folder_path):
        print(f"Directory {folder_path} not found.")
        sys.exit(1)
        
    case_id = os.path.basename(os.path.normpath(folder_path))
    print(f"Loading data for case: {case_id}")
    
    modalities = ['t1', 't1ce', 't2', 'flair', 'seg']
    data_dict = {}
    
    for mod in modalities:
        search_pattern = os.path.join(folder_path, f"*_{mod}.nii.gz")
        files = glob.glob(search_pattern)
        if files:
            file_path = files[0]
            data = load_nifti(file_path)
            if data is not None:
                if mod != 'seg':
                    data = normalize(data)
                data_dict[mod] = data
                print(f"Loaded {mod}: shape {data.shape}")
        else:
            print(f"Warning: Modality {mod} not found.")

    if not data_dict:
        print("No valid NIfTI files found in the directory.")
        sys.exit(1)
        
    # Assume all loaded volumes have the same shape
    ref_mod = list(data_dict.keys())[0]
    data_shape = data_dict[ref_mod].shape
    
    initial_axis = 2  # default to axial (z-axis)
    init_slice = data_shape[initial_axis] // 2
    
    # Setup the plot grid based on how many modalities we loaded
    num_plots = len(data_dict)
    
    # Custom colormap for segmentation overlay
    # 0 = background (transparent), 1 = necrosis (red), 2 = edema (green), 4 = enhancing tumor (yellow)
    # Note: BraTS typically has labels 1, 2, 4. We map them appropriately.
    # We will just show segmentation as a distinct categorical map.
    cmap_seg = ListedColormap(['black', 'red', 'green', 'blue', 'yellow'])
    
    fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    plt.subplots_adjust(bottom=0.25, wspace=0.1)
    
    def get_slice(data, axis, index):
        if axis == 0:
            slice_data = data[index, :, :]
        elif axis == 1:
            slice_data = data[:, index, :]
        else:
            slice_data = data[:, :, index]
        return np.rot90(slice_data)

    im_dict = {}
    mod_keys = list(data_dict.keys())
    
    for i, mod in enumerate(mod_keys):
        ax = axes[i]
        slice_data = get_slice(data_dict[mod], initial_axis, init_slice)
        
        if mod == 'seg':
            # Force vmin/vmax for standard BraTS labels if possible (0, 1, 2, 4)
            im = ax.imshow(slice_data, cmap=cmap_seg, vmin=0, vmax=4, interpolation='nearest')
        else:
            im = ax.imshow(slice_data, cmap="gray", vmin=0, vmax=1)
            
        ax.set_title(mod.upper())
        ax.axis('off')
        im_dict[mod] = im

    # Add slider
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    max_slice = data_shape[initial_axis] - 1
    slice_slider = Slider(
        ax=ax_slider,
        label='Slice Index',
        valmin=0,
        valmax=max_slice,
        valinit=init_slice,
        valstep=1
    )

    # Add radio buttons for axis selection
    ax_radio = plt.axes([0.05, 0.05, 0.15, 0.12])
    radio = RadioButtons(ax_radio, ('Sagittal (X)', 'Coronal (Y)', 'Axial (Z)'), active=2)

    def update(val):
        idx = int(slice_slider.val)
        current_axis_label = radio.value_selected
        ax_map = {'Sagittal (X)': 0, 'Coronal (Y)': 1, 'Axial (Z)': 2}
        a = ax_map[current_axis_label]
        
        for i, mod in enumerate(mod_keys):
            slice_data = get_slice(data_dict[mod], a, idx)
            im_dict[mod].set_data(slice_data)
            
        fig.suptitle(f"Case {case_id} | {current_axis_label} - Slice: {idx}", fontsize=14)
        fig.canvas.draw_idle()

    def update_axis(label):
        ax_map = {'Sagittal (X)': 0, 'Coronal (Y)': 1, 'Axial (Z)': 2}
        a = ax_map[label]
        
        new_max = data_shape[a] - 1
        slice_slider.valmax = new_max
        slice_slider.ax.set_xlim(0, new_max)
        
        if slice_slider.val > new_max:
            slice_slider.set_val(new_max)
        else:
            update(slice_slider.val)

    # Initialize title
    fig.suptitle(f"Case {case_id} | Axial (Z) - Slice: {init_slice}", fontsize=14)

    slice_slider.on_changed(update)
    radio.on_clicked(update_axis)

    plt.show()

if __name__ == "__main__":
    main()
