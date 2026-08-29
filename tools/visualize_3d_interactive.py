import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import nibabel as nib

def main():
    parser = argparse.ArgumentParser(description="Visualize and interact with 3D NIfTI (.nii or .nii.gz) data.")
    parser.add_argument("file_path", type=str, nargs='?', 
                        default="datasets/brats/BraTS2021_00186/BraTS2021_00186_t1ce.nii.gz",
                        help="Path to the .nii or .nii.gz file")
    
    args = parser.parse_args()
    file_path = args.file_path
    
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        print(f"Loaded {file_path}")
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"Failed to load {file_path}. Error: {e}")
        sys.exit(1)
        
    if len(data.shape) != 3 and len(data.shape) != 4:
        print("This script is designed for 3D data. Data shape is not 3D.")
        sys.exit(1)

    # If it's 4D (e.g. multi-channel), just take the first channel
    if len(data.shape) == 4:
        print("Warning: 4D data detected. Displaying the first volume only.")
        data = data[..., 0]

    # Normalize data for better visualization
    p1 = np.percentile(data[data > 0], 1) if np.any(data > 0) else 0
    p99 = np.percentile(data[data > 0], 99) if np.any(data > 0) else np.max(data)
    data = np.clip(data, p1, p99)
    if p99 > p1:
        data = (data - p1) / (p99 - p1)

    # Initial setup
    initial_axis = 2  # default to axial (z-axis)
    init_slice = data.shape[initial_axis] // 2
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25, left=0.25) # Adjust layout to make room for widgets
    
    # helper to get 2d slice
    def get_slice(data, axis, index):
        if axis == 0:
            slice_data = data[index, :, :]
        elif axis == 1:
            slice_data = data[:, index, :]
        else:
            slice_data = data[:, :, index]
        
        # Rotate for standard orientation in matplotlib
        return np.rot90(slice_data)

    im = ax.imshow(get_slice(data, initial_axis, init_slice), cmap="gray")
    ax.set_title(f"Axial (Z) - Slice {init_slice}")
    ax.axis('off')

    # Add slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    max_slice = data.shape[initial_axis] - 1
    slice_slider = Slider(
        ax=ax_slider,
        label='Slice Index',
        valmin=0,
        valmax=max_slice,
        valinit=init_slice,
        valstep=1
    )

    # Add radio buttons for axis selection
    ax_radio = plt.axes([0.02, 0.4, 0.18, 0.15])
    radio = RadioButtons(ax_radio, ('Sagittal (X)', 'Coronal (Y)', 'Axial (Z)'), active=2)

    def update(val):
        idx = int(slice_slider.val)
        current_axis_label = radio.value_selected
        ax_map = {'Sagittal (X)': 0, 'Coronal (Y)': 1, 'Axial (Z)': 2}
        a = ax_map[current_axis_label]
        
        im.set_data(get_slice(data, a, idx))
        ax.set_title(f"{current_axis_label} - Slice: {idx}")
        fig.canvas.draw_idle()

    def update_axis(label):
        ax_map = {'Sagittal (X)': 0, 'Coronal (Y)': 1, 'Axial (Z)': 2}
        a = ax_map[label]
        
        # update slider min/max 
        new_max = data.shape[a] - 1
        slice_slider.valmax = new_max
        slice_slider.ax.set_xlim(0, new_max)
        
        # keep index clamped
        if slice_slider.val > new_max:
            slice_slider.set_val(new_max)
        else:
            # Force update to redraw with new axis and current val
            update(slice_slider.val)

    slice_slider.on_changed(update)
    radio.on_clicked(update_axis)

    plt.show()

if __name__ == "__main__":
    main()
