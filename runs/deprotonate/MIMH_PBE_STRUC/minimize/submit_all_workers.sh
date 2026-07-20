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
    LIMIT=$((QM_COUNT - 1))
    echo "Submitting $QM_COUNT workers for QM methods (8 cores, array 0-$LIMIT)..."
    sbatch --array=0-$LIMIT worker_qm.slurm
else
    echo "No tasks for QM methods."
fi

# 4. Submit Other Workers (max {MAX_OTHER_WORKERS})
if [ "$OTHER_COUNT" -gt 0 ]; then
    LIMIT=$((OTHER_COUNT - 1))
    echo "Submitting $OTHER_COUNT workers for other runs (1 core, array 0-$LIMIT)..."
    sbatch --array=0-$LIMIT worker_other.slurm
else
    echo "No tasks for other runs."
fi
