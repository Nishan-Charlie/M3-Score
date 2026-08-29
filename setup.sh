#!/bin/bash

# Exit on error
set -e

echo "Starting installation for MRI Diffusion codebase..."

# 1. Initialize Conda for this shell session
# This finds where conda is installed and sources the activation logic
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

# 2. Create and Activate Environment
# We use -y to avoid the manual [Y/n] prompt during script execution
conda create -n mri-diffuser python=3.10 -y
echo "New conda environment created: mri-diffuser"

conda activate mri-diffuser
echo "Conda environment activated: mri-diffuser"

# 3. Install Torch with CUDA 12.4 support
echo "Installing Torch, Torchvision, and Torchaudio with CUDA 12.4 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install the rest of the requirements
if [ -f "requirements.txt" ]; then
    echo "Installing remaining requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Installing core dependencies individually..."
    # Added 'torchmetrics' and 'scikit-image' for your SOTA metrics
    pip install monai diffusers accelerate tqdm pandas numpy scipy nibabel matplotlib torchmetrics[image] scikit-image
fi

# 5. Initialize Accelerate (Optional but recommended for FP16)
# This creates a default config so you don't have to manually 'accelerate config'
mkdir -p ~/.cache/huggingface/accelerate
cat <<EOT > ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: fp16
use_cpu: false
EOT

echo "------------------------------------------------"
echo "Installation Complete!"
echo "To use this environment in the future, run:"
echo "conda activate mri-diffuser"
echo "------------------------------------------------"