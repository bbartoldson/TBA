#!/bin/bash

config="multi_gpu_tba_rho1b_gsm8k"

# Setup
num_trainer_processes=2 # this is hard-coded. changing it requires changes to visible devices and the python code.

num_processes=4  # 2 trainers + 2 searchers
#num_processes=8  # 2 trainers + 6 searchers

LAUNCHER="srun -G 4 -N 1 -n ${num_processes}"
#LAUNCHER="srun -G 8 -N 2 -n ${num_processes}"

USE_DEEPSPEED=true  # Set to false to disable DeepSpeed
USE_DEEPSPEED=false  # Set to false to disable DeepSpeed
# Setup Finished

PYTHON_CMD="python tba_gsm8k.py --config configs/${config}.yml \
    --run_name multi_gsm8k_run_0 \
    --output_dir multi_gsm8k_run_0"

# Add DeepSpeed arguments if enabled
if [ "$USE_DEEPSPEED" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --use_deepspeed True"
    echo "📚 DeepSpeed enabled"
fi

echo "🔧 Process configuration:"
echo "  - Total processes: $num_processes"
echo "  - Trainer processes: $num_trainer_processes (ranks 0,1,...)"
echo "  - Searcher processes: $((num_processes - num_trainer_processes)) (ranks 2,3,...)"
echo ""

echo "Using SLURM (srun)..."

# Execute with proper GPU assignment
$LAUNCHER bash -c "
    # Determine launcher type and get rank
    if [ -n \"\$SLURM_PROCID\" ]; then
        RANK=\$SLURM_PROCID
    else
        echo \"❌ Cannot determine process rank\"
        exit 1
    fi
    
    # Set GPU visibility based on role
    if [ \$RANK -lt $num_trainer_processes ]; then
        export CUDA_VISIBLE_DEVICES=0,1
        ROLE=\"trainer\"
    else
        export CUDA_VISIBLE_DEVICES=\$((RANK % 4))
        ROLE=\"searcher\"
    fi
    
    echo \"Process \$RANK (\$ROLE): CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"
    
    # Execute the command
    $PYTHON_CMD
"

echo ""
echo "✅ Launch completed!"
