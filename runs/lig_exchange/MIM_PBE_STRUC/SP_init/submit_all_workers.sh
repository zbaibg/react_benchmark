#!/bin/bash

# 1. Distribute tasks using Python script
echo "Distributing tasks..."

# 2. Read the job counts
if [ -f "task_lists/job_counts.txt" ]; then
    source task_lists/job_counts.txt
else
    echo "Error: task_lists/job_counts.txt not generated."
    exit 1
fi

# 3. Submit QM Workers (max {MAX_QM_WORKERS})
if [ "$QM_COUNT" -gt 0 ]; then
    echo "Submitting $QM_COUNT workers for QM methods (16 cores)..."
    for i in $(seq 0 $((QM_COUNT - 1))); do
        sbatch --spread-job \
            --output=slurm_logs/worker_qm_%j_${i}.out \
            --error=slurm_logs/worker_qm_%j_${i}.err \
            worker_qm.slurm $i
    done
else
    echo "No tasks for QM methods."
fi

# 4. Submit Other Workers (max {MAX_OTHER_WORKERS})
if [ "$OTHER_COUNT" -gt 0 ]; then
    echo "Submitting $OTHER_COUNT workers for other runs (6 core)..."
    for i in $(seq 0 $((OTHER_COUNT - 1))); do
        sbatch --spread-job \
            --output=slurm_logs/worker_other_%j_${i}.out \
            --error=slurm_logs/worker_other_%j_${i}.err \
            worker_other.slurm $i
    done
else
    echo "No tasks for other runs."
fi
