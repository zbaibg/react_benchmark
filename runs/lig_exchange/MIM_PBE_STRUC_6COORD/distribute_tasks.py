#!/usr/bin/env python3
import os
import sys
import glob
import math
import argparse
import shutil
# ============ Configuration ============
QM_RUNS = ["run10", "run11", 'run12','run13','run14','run16','run24']  # QM method run directories
MAX_QM_WORKERS = 40           # Maximum number of QM workers
MAX_OTHER_WORKERS = 300       # Maximum number of other workers
QM_CPUS_PER_TASK = 16          # CPUs per task for QM workers
OTHER_CPUS_PER_TASK = 1       # CPUs per task for other workers
# =======================================


def chunk_list(data, num_chunks):
    """Distribute tasks as evenly as possible"""
    avg = len(data) / float(num_chunks)
    out = []
    last = 0.0
    while last < len(data):
        out.append(data[int(last):int(last + avg)])
        last += avg
    while len(out) < num_chunks:
        out.append([])
    if len(out) > num_chunks:
        last_chunk = out[num_chunks-1]
        for i in range(num_chunks, len(out)):
            last_chunk.extend(out[i])
        out = out[:num_chunks]
    return out

def write_task_files(target_dir, tasks, prefix, max_workers):
    """Write task lists to files.

    We create at most `max_workers` task files, with *contiguous* indices
    from 0 to N-1 so that Slurm array 0-(N-1) always finds a file.
    """
    if not tasks:
        return 0

    # Decide how many workers to actually use
    num_workers = min(max_workers, len(tasks))

    task_lists_dir = os.path.join(target_dir, "task_lists")
    os.makedirs(task_lists_dir, exist_ok=True)

    # Evenly split tasks into `num_workers` chunks
    chunk_size = math.ceil(len(tasks) / float(num_workers))
    count = 0
    for i in range(num_workers):
        start = int(i * chunk_size)
        end = int(min((i + 1) * chunk_size, len(tasks)))
        chunk = tasks[start:end]
        if not chunk:
            break
        filename = os.path.join(task_lists_dir, f"{prefix}_{i}.txt")
        with open(filename, "w") as f:
            for task in chunk:
                f.write(task + "\n")
        count += 1
    return count

def generate_submit_script(target_dir, script_name):
    """Generate submit_all_workers.sh"""
    # Construct the python command with the script argument if it's not the default
    cmd_args = ""
    if script_name != "sbatch_prepare.sh":
        cmd_args = f" --script {script_name}"
        
    script_content = f"""#!/bin/bash

# 1. Distribute tasks using Python script
echo "Distributing tasks..."

# 2. Read the job counts
if [ -f "task_lists/job_counts.txt" ]; then
    source task_lists/job_counts.txt
else
    echo "Error: task_lists/job_counts.txt not generated."
    exit 1
fi

# 3. Submit QM Workers (max {{MAX_QM_WORKERS}})
if [ "$QM_COUNT" -gt 0 ]; then
    LIMIT=$((QM_COUNT - 1))
    echo "Submitting $QM_COUNT workers for QM methods (8 cores, array 0-$LIMIT)..."
    sbatch --array=0-$LIMIT worker_qm.slurm
else
    echo "No tasks for QM methods."
fi

# 4. Submit Other Workers (max {{MAX_OTHER_WORKERS}})
if [ "$OTHER_COUNT" -gt 0 ]; then
    LIMIT=$((OTHER_COUNT - 1))
    echo "Submitting $OTHER_COUNT workers for other runs (1 core, array 0-$LIMIT)..."
    sbatch --array=0-$LIMIT worker_other.slurm
else
    echo "No tasks for other runs."
fi
"""
    filepath = os.path.join(target_dir, "submit_all_workers.sh")
    with open(filepath, "w") as f:
        f.write(script_content)
    os.chmod(filepath, 0o755)
    print(f"Generated: {filepath}")

def generate_worker_qm_slurm(target_dir, script_name):
    """Generate worker_qm.slurm"""
    log_base = os.path.splitext(script_name)[0]
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=w_qm
#SBATCH --output=slurm_logs/worker_qm_%A_%a.out
#SBATCH --error=slurm_logs/worker_qm_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={QM_CPUS_PER_TASK}
#SBATCH --mem-per-cpu=8G
#SBATCH --exclude=compute-0-[0-40,44]

TASK_LIST="task_lists/qm_${{SLURM_ARRAY_TASK_ID}}.txt"

if [ ! -f "$TASK_LIST" ]; then
    echo "Task list $TASK_LIST not found!"
    exit 0
fi

echo "Processing tasks from $TASK_LIST"
while IFS= read -r SCRIPT_PATH; do
    if [ -z "$SCRIPT_PATH" ]; then continue; fi
    
    DIR=$(dirname "$SCRIPT_PATH")
    echo "Starting task in $DIR"
    
    (
        cd "$DIR" || exit
        # Redirect stdin to /dev/null so the inner script doesn't eat the loop input
        bash {script_name} > {log_base}.log 2> {log_base}.err < /dev/null
    )
    
done < "$TASK_LIST"
echo "All tasks in list completed."
"""
    filepath = os.path.join(target_dir, "worker_qm.slurm")
    with open(filepath, "w") as f:
        f.write(slurm_content)
    print(f"Generated: {filepath}")

def generate_worker_other_slurm(target_dir, script_name):
    """Generate worker_other.slurm"""
    log_base = os.path.splitext(script_name)[0]
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=w_other
#SBATCH --output=slurm_logs/worker_other_%A_%a.out
#SBATCH --error=slurm_logs/worker_other_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={OTHER_CPUS_PER_TASK}
#SBATCH --mem-per-cpu=8G
#SBATCH --exclude=compute-0-[0-40,44]

TASK_LIST="task_lists/other_${{SLURM_ARRAY_TASK_ID}}.txt"

if [ ! -f "$TASK_LIST" ]; then
    echo "Task list $TASK_LIST not found!"
    exit 0
fi

echo "Processing tasks from $TASK_LIST"
while IFS= read -r SCRIPT_PATH; do
    if [ -z "$SCRIPT_PATH" ]; then continue; fi
    
    DIR=$(dirname "$SCRIPT_PATH")
    echo "Starting task in $DIR"
    
    (
        cd "$DIR" || exit
        # Redirect stdin to /dev/null so the inner script doesn't eat the loop input
        bash {script_name} > {log_base}.log 2> {log_base}.err < /dev/null
    )
    
done < "$TASK_LIST"
echo "All tasks in list completed."
"""
    filepath = os.path.join(target_dir, "worker_other.slurm")
    with open(filepath, "w") as f:
        f.write(slurm_content)
    print(f"Generated: {filepath}")

def process_single_directory(target_subdir, script_dir, script_name):
    """Process a single target directory"""
    target_dir = os.path.join(script_dir, target_subdir)
    
    if not os.path.exists(target_dir):
        print(f"Error: Target directory {target_dir} does not exist!")
        return False

    if os.path.exists(os.path.join(target_dir, "task_lists")):
        print(f"Task lists already exist in {target_dir}, will delete it first")
        shutil.rmtree(os.path.join(target_dir, "task_lists"))
    os.makedirs(os.path.join(target_dir, "task_lists"))
    if os.path.exists(os.path.join(target_dir, "slurm_logs")):
        print(f"Slurm logs already exist in {target_dir}, will delete it first")
        shutil.rmtree(os.path.join(target_dir, "slurm_logs"))
    os.makedirs(os.path.join(target_dir, "slurm_logs"))

    # Save current directory
    original_dir = os.getcwd()

    try:
        # Change to target directory to find tasks
        os.chdir(target_dir)
        
        # Collect QM method tasks
        qm_tasks = []
        qm_missing = []
        if script_name != "prepare.sh":
            for qm_run in QM_RUNS:
                tasks = glob.glob(f"./{qm_run}/*/{script_name}")
                qm_tasks.extend(tasks)
                
                # Check for subdirectories without the script
                run_path = f"./{qm_run}"
                if os.path.exists(run_path):
                    for subdir in glob.glob(os.path.join(run_path, "*")):
                        if os.path.isdir(subdir):
                            sbatch_file = os.path.join(subdir, script_name)
                            if not os.path.exists(sbatch_file):
                                # Check if prepare.sh exists (indicates this is a valid structure dir)
                                # Only check prepare.sh if we are not already looking for prepare.sh
                                if script_name != "prepare.sh":
                                    prepare_file = os.path.join(subdir, "prepare.sh")
                                    if os.path.exists(prepare_file):
                                        qm_missing.append(os.path.relpath(subdir))
                                else:
                                    # If we are looking for prepare.sh, then it's missing
                                    qm_missing.append(os.path.relpath(subdir))

        qm_tasks.sort()
        
        # Collect tasks from other runs
        all_run_dirs = glob.glob("./run*")
        other_tasks = []
        other_missing = []
        for run_dir in all_run_dirs:
            run_name = os.path.basename(run_dir)
            if run_name in QM_RUNS and script_name != "prepare.sh":
                continue
            tasks = glob.glob(os.path.join(run_dir, "*", script_name))
            other_tasks.extend(tasks)
            
            # Check for subdirectories without the script
            for subdir in glob.glob(os.path.join(run_dir, "*")):
                if os.path.isdir(subdir):
                    sbatch_file = os.path.join(subdir, script_name)
                    if not os.path.exists(sbatch_file):
                        # Check if prepare.sh exists (indicates this is a valid structure dir)
                        if script_name != "prepare.sh":
                            prepare_file = os.path.join(subdir, "prepare.sh")
                            if os.path.exists(prepare_file):
                                other_missing.append(os.path.relpath(subdir))
                        else:
                             # If we are looking for prepare.sh, then it's missing
                             # But we need a check to see if it's a valid dir? 
                             # Assume subdirs in run* are valid if they are directories
                             other_missing.append(os.path.relpath(subdir))
        
        other_tasks.sort()
        
        print(f"\\n{'='*60}")
        print(f"Processing directory: {target_subdir}")
        print(f"Target script: {script_name}")
        print(f"{'='*60}")
        print(f"Found {len(qm_tasks)} tasks for QM methods ({{' + '.join(QM_RUNS)}}).")
        print(f"Found {len(other_tasks)} tasks for other runs.")
        
        # Print warnings for missing script files
        if qm_missing:
            print(f"\\n⚠️  Warning: {len(qm_missing)} QM subdirectories missing {script_name}:")
            for missing_dir in qm_missing:
                print(f"   - {missing_dir}")
            if script_name != "prepare.sh":
                print(f"   (Run 'bash prepare.sh' in these directories to generate sbatch files)")
        
        if other_missing:
            print(f"\\n⚠️  Warning: {len(other_missing)} other subdirectories missing {script_name}:")
            for missing_dir in other_missing:
                print(f"   - {missing_dir}")
            if script_name != "prepare.sh":
                print(f"   (Run 'bash prepare.sh' in these directories to generate sbatch files)")
        
        # Ensure task_lists directory exists
        task_lists_dir = os.path.join(target_dir, "task_lists")
        os.makedirs(task_lists_dir, exist_ok=True)
        
        # Write task files
        n_qm = write_task_files(target_dir, qm_tasks, "qm", MAX_QM_WORKERS)
        print(f"Generated {n_qm} task lists for QM methods (max {MAX_QM_WORKERS}).")
        
        n_other = write_task_files(target_dir, other_tasks, "other", MAX_OTHER_WORKERS)
        print(f"Generated {n_other} task lists for other runs (max {MAX_OTHER_WORKERS}).")
        
        # Write job counts file
        job_counts_file = os.path.join(task_lists_dir, "job_counts.txt")
        with open(job_counts_file, "w") as f:
            f.write(f"QM_COUNT={n_qm}\n")
            f.write(f"OTHER_COUNT={n_other}\n")
        print(f"Generated: {job_counts_file}")
        
        # Generate slurm scripts and submit script
        generate_submit_script(target_dir, script_name)
        generate_worker_qm_slurm(target_dir, script_name)
        generate_worker_other_slurm(target_dir, script_name)
        
        print(f"\\n✓ All files generated in {target_dir}")
        print(f"  - submit_all_workers.sh")
        print(f"  - worker_qm.slurm (QM methods: {{', '.join(QM_RUNS)}})")
        print(f"  - worker_other.slurm (Other runs)")
        print(f"  - task_lists/ (with {n_qm + n_other} task files)")
        print(f"\\nTo submit jobs, run:")
        print(f"  cd {target_subdir} && bash submit_all_workers.sh")
        
        return True
        
    finally:
        # Restore original directory
        os.chdir(original_dir)

def main():
    parser = argparse.ArgumentParser(description="Distribute tasks for QM/MM runs.")
    parser.add_argument("target_subdirs", nargs="+", help="Target directories to process")
    parser.add_argument("--script", "-s", default="sbatch_prepare.sh", help="Script to run (default: sbatch_prepare.sh)")
    
    args = parser.parse_args()
    target_subdirs = args.target_subdirs
    script_name = args.script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\\n{'#'*60}")
    print(f"# Processing {len(target_subdirs)} director{{'y' if len(target_subdirs) == 1 else 'ies'}}")
    print(f"Using script: {script_name}")
    print(f"{'#'*60}")
    
    success_count = 0
    failed_dirs = []
    
    for target_subdir in target_subdirs:
        if process_single_directory(target_subdir, script_dir, script_name):
            success_count += 1
        else:
            failed_dirs.append(target_subdir)
    
    # Summary
    print(f"\\n{'#'*60}")
    print(f"# Summary")
    print(f"{'#'*60}")
    print(f"Successfully processed: {success_count}/{len(target_subdirs)} directories")
    
    if failed_dirs:
        print(f"\\nFailed directories:")
        for failed_dir in failed_dirs:
            print(f"  - {failed_dir}")
        sys.exit(1)
    else:
        print(f"\\n✓ All directories processed successfully!")
        print(f"\\nTo submit all jobs, run:")
        for target_subdir in target_subdirs:
            print(f"  cd {target_subdir} && bash submit_all_workers.sh && cd ..")

if __name__ == "__main__":
    main()
