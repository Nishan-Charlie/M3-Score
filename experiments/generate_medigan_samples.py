import os
import shutil
from medigan import Generators

def main():
    print("Initializing Medigan Generators...")
    generators = Generators()

    # Define the modalities we want to generate
    modalities = ['Mammography', 'X-Ray', 'MRI']
    
    output_base_dir = "medigan_samples"
    # Clean up previous attempts to avoid confusion
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    
    num_samples = 100

    for modality in modalities:
        print(f"\nSearching for {modality} models...")
        # Get list of model IDs for this modality
        model_ids = generators.get_models_by_key_value_pair(key1="modality", value1=modality)
        
        if not model_ids:
            print(f"No models found for modality: {modality}")
            continue
            
        # Select the first available model for this modality
        selected_model_dict = model_ids[0]
        selected_model = selected_model_dict['model_id'] if isinstance(selected_model_dict, dict) else selected_model_dict
        print(f"Selected Model ID: {selected_model}")
        
        output_dir = os.path.join(output_base_dir, f"{modality}_{selected_model}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {num_samples} samples into {output_dir}...")
        
        try:
            # Generate the images
            generators.generate(
                model_id=selected_model,
                num_samples=num_samples,
                output_path=output_dir,
                save_images=True,
                is_gen_disable_sm=True, # Disable metrics computation during generation
                install_dependencies=True # Automatically handle missing dependencies
            )
            print(f"✅ Successfully generated {num_samples} {modality} images.")
        except Exception as e:
            print(f"❌ Failed to generate images for {selected_model}: {str(e)}")

if __name__ == "__main__":
    main()
