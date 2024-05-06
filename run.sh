#!/bin/bash -l
#SBATCH --job-name=NetworkEnv
#SBATCH --output=script_logs/%x_%j_output.log
#SBATCH --error=script_logs/%x_%j_error.log
#SBATCH -G 1 -C a4500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sj1030@rutgers.edu

echo "Starting job $SLURM_JOB_ID"
set -e

cd ~/Projects/swarm-network-coverage

source /common/home/sj1030/miniconda3/etc/profile.d/conda.sh

source ~/.bashrc
conda activate rl || { echo "Failed to activate conda environment"; exit 1; }

gpustat

START_TIME=$(date +%s)

# Function to print time elapsed every 5 minutes
print_time_elapsed() {
    while :
    do
        # Get the current time
        CURRENT_TIME=$(date +%s)
        
        # Calculate the time elapsed
        TIME_ELAPSED=$((CURRENT_TIME - START_TIME))
        
        # Convert the time elapsed to hours, minutes, and seconds
        HOURS=$((TIME_ELAPSED / 3600))
        MINUTES=$(( (TIME_ELAPSED % 3600) / 60 ))
        SECONDS=$((TIME_ELAPSED % 60))
        
        # Print the time elapsed
        echo "Time elapsed: $HOURS hours, $MINUTES minutes, $SECONDS seconds"
        
        # Print GPU usage
        gpustat
        
        # Sleep for 5 minutes
        sleep 120
    done
}

# Start the time elapsed function in the background
print_time_elapsed &

# Save the process ID of the background function
TIME_ELAPSED_PID=$!

python run.py > script_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_python.log 2>&1 &

echo $!
wait $!

gpustat

# Kill the time elapsed function
kill $TIME_ELAPSED_PID
