#!/bin/bash

#SBATCH --job-name=alex_a100_4_1024
#SBATCH --output=logs/alex_a100_4_1024_%j.out
#SBATCH --error=logs/alex_a100_4_1024_%j.err
#SBATCH --partition=kempner
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # Number of processes (1 per GPU)
#SBATCH --gres=gpu:4             # Total GPUs per node
#SBATCH --cpus-per-task=16       # Number of CPU cores per task
#SBATCH --mem=244GB              # Memory per node
#SBATCH --time=36:00:00          # Maximum runtime
#SBATCH --constraint=a100


# User and Envrironment Variables ----------------------------------------------

# Starting the changing and customizing the script from here ------

# User Variables ---------------
ARCHITECTURE=alexnet  # Model architecture (e.g., resnet50)
EPOCHS=90              # Number of epochs to train
BATCH_SIZE=1024         # Total batch size 
LEARNING_RATE=0.1      # Initial learning rate
WORKERS=16             # Number of CPU workers for data loading

# Environment Variables --------
# Environment Variables --------
PROJECT_DIR=/n/netscratch/kempner_dev/Lab/bdesinghu/Vision/AlexNet/
CONDA_ENV=/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/pytorch_image # <--- Conda environment path


# End of the changing and customizing the script ------------------
# DON'T CHANGE ANYTHING BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING
#-------------------------------------------------------------------------------


PYTHON_BINARY=$CONDA_ENV/bin/python
DATA_DIR="/n/holylfs06/LABS/kempner_shared/Lab/data/imagenet_1k"
SCRIPT_DIR=$PROJECT_DIR/scripts/main.py


echo "----------------------- *** -----------------------"
echo "Project Directory: $PROJECT_DIR"
echo "Conda Environment: $CONDA_ENV"
echo "Python Binary: $PYTHON_BINARY"
echo "Data Directory: $DATA_DIR"
echo "----------------------- *** -----------------------"

# Select a free port -----------------------------------------------------------
get_free_port() {
    while true; do
        # Choose a random port between 20000 and 30000
        PORT=$(shuf -i 20000-30000 -n 1)
        # Check if the port is free using ss or netstat
        if ! ss -lpn | grep -q ":$PORT "; then
            # If the port is not in use, break the loop and return the port
            echo $PORT
            break
        fi
    done
}

# load modules -----------------------------------------------------------------
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

# Activate conda environment (although I use full path because sometimes
# the conda environment is not activated properly)

conda activate $CONDA_ENV

# Set environment variables for PyTorch distributed training
export MASTER_PORT=$(get_free_port)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

# Print the environment variables
echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST
echo "MASTER_ADDR = "$MASTER_ADDR
echo "MASTER_PORT = "$MASTER_PORT
echo "WORLD_SIZE = "$WORLD_SIZE

start_time=$(date +%s)
echo "Start time: $(date)"

CMD="srun $PYTHON_BINARY \
    $SCRIPT_DIR \
    -a $ARCHITECTURE \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --workers $WORKERS \
    --dist-url tcp://$MASTER_ADDR:$MASTER_PORT \
    --dist-backend nccl \
    --multiprocessing-distributed \
    --world-size $WORLD_SIZE --rank $SLURM_PROCID \
    $DATA_DIR"

echo "$CMD"
eval "$CMD"


end_time=$(date +%s)
duration=$((end_time - start_time))
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo " Total duration: $duration seconds."

