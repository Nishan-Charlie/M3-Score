import gradio as gr
import nibabel as nib
import numpy as np
import argparse
import sys
import os
import glob

def get_available_patients(data_dir="datasets/brats"):
    patients = []
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, item)) and item.startswith("BraTS2021_"):
                patients.append(item)
    return sorted(patients)

def load_data(file_path):
    print(f"Loading {file_path}")
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        if len(data.shape) == 4:
            data = data[..., 0]
        
        # Normalize
        mask = data > 0
        if np.any(mask):
            p1 = np.percentile(data[mask], 1)
            p99 = np.percentile(data[mask], 99)
            data = np.clip(data, p1, p99)
            if p99 > p1:
                data = (data - p1) / (p99 - p1)
                
        # Scale to 0-255 for image display
        data = (data * 255).astype(np.uint8)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="datasets/brats", help="Path to the brats dataset folder")
    args = parser.parse_args()
    
    patients = get_available_patients(args.data_dir)
    modalities = ["flair", "t1", "t1ce", "t2", "seg"]
    
    if not patients:
        print(f"Warning: No patients found in {args.data_dir}. Check the data_dir path.")
        patients = ["BraTS2021_00186"] # Fallback to prevent crash

    def load_new_volume(patient_id, modality):
        if not patient_id:
            return None, gr.Slider(maximum=1, value=0), None, "Please select a patient."
            
        file_path = os.path.join(args.data_dir, patient_id, f"{patient_id}_{modality}.nii.gz")
        if not os.path.exists(file_path):
            return None, gr.Slider(maximum=1, value=0), None, f"File not found: {file_path}"
            
        data = load_data(file_path)
        if data is None:
            return None, gr.Slider(maximum=1, value=0), None, f"Failed to load {file_path}"
        
        # Default to Axial max
        m = data.shape[2] - 1
        return data, gr.Slider(minimum=0, maximum=m, value=m//2), get_slice_from_data(data, "Axial (Z)", m//2), f"Loaded {file_path}"

    def get_slice_from_data(data, axis, slice_idx):
        if data is None:
            return np.zeros((256, 256))
        try:
            if axis == "Axial (Z)":
                img_slice = data[:, :, int(slice_idx)]
            elif axis == "Coronal (Y)":
                img_slice = data[:, int(slice_idx), :]
            else: # Sagittal (X)
                img_slice = data[int(slice_idx), :, :]
                
            img_slice = np.rot90(img_slice)
            return img_slice
        except Exception as e:
            return np.zeros((256, 256))

    def update_max_slider(data, axis):
        if data is None:
            return gr.Slider(maximum=1, value=0)
        try:
            if axis == "Axial (Z)":
                m = data.shape[2] - 1
            elif axis == "Coronal (Y)":
                m = data.shape[1] - 1
            else:
                m = data.shape[0] - 1
            return gr.Slider(minimum=0, maximum=m, value=m//2, step=1, label="Slice Index", interactive=True)
        except:
             return gr.Slider(maximum=1, value=0)

    with gr.Blocks(title="3D MRI Viewer") as demo:
        gr.Markdown(f"# 3D MRI Dataset Viewer")
        data_state = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Select File")
                patient_dropdown = gr.Dropdown(choices=patients, value=patients[0], label="Patient ID", interactive=True)
                modality_dropdown = gr.Dropdown(choices=modalities, value="t1ce", label="Modality", interactive=True)
                load_btn = gr.Button("Load Data", variant="primary")
                status_text = gr.Markdown("Ready.")
                
                gr.Markdown("### 2. View Options")
                axis_radio = gr.Radio(["Axial (Z)", "Coronal (Y)", "Sagittal (X)"], value="Axial (Z)", label="View Axis")
                slice_slider = gr.Slider(0, 150, value=75, step=1, label="Slice Index")
            with gr.Column(scale=3):
                image_output = gr.Image(label="MRI Slice", image_mode="L", type="numpy", height=700)
                
        # When Load Data is clicked
        load_btn.click(
            fn=load_new_volume,
            inputs=[patient_dropdown, modality_dropdown],
            outputs=[data_state, slice_slider, image_output, status_text]
        )
        
        # When axis changes, update the slider's valid range and max, and the image
        axis_radio.change(
            fn=update_max_slider, 
            inputs=[data_state, axis_radio], 
            outputs=slice_slider
        ).then(
            fn=get_slice_from_data, 
            inputs=[data_state, axis_radio, slice_slider], 
            outputs=image_output
        )
        
        # When slider changes, update the image
        slice_slider.change(
            fn=get_slice_from_data, 
            inputs=[data_state, axis_radio, slice_slider], 
            outputs=image_output
        )

        # Initial load
        demo.load(
            fn=load_new_volume,
            inputs=[patient_dropdown, modality_dropdown],
            outputs=[data_state, slice_slider, image_output, status_text]
        )

    # Launch server
    demo.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    main()

