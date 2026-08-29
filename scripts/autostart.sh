#!/bin/bash

# Usage: ./autostart.sh <REQUIRED_VRAM_MB> <GPU_ID> <COMMAND>
# Example: ./autostart.sh 10000 0 "./run_pipeline.sh"

REQUIRED_VRAM_MB=${1:-20000}
GPU_ID=${2:-0}
COMMAND=${3:-"nohup bash ./run_pipeline.sh &"}
CHECK_INTERVAL=60 # seconds

echo "Waiting for ${REQUIRED_VRAM_MB} MB of free VRAM on GPU ${GPU_ID}..."
echo "Command to run: ${COMMAND}"

while true; do
    # Get free VRAM in MB for the specific GPU
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID)
    
    if [ "$FREE_VRAM" -ge "$REQUIRED_VRAM_MB" ]; then
        echo "$(date): Sufficient VRAM available ($FREE_VRAM MB). Starting the pipeline..."
        # Execute the command
        eval "$COMMAND"
        break
    else
        echo "$(date): Only $FREE_VRAM MB available. Need $REQUIRED_VRAM_MB MB. Checking again in $CHECK_INTERVAL seconds..."
        sleep $CHECK_INTERVAL
    fi
done
