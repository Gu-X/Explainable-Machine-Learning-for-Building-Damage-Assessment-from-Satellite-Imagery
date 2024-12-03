def generate_submit_file(args):
    dataset_paths = {
        0: "/vol/research/ak0084_datasets/Datasets/massachusetts-buildings-dataset",
        1: "/vol/research/ak0084_datasets/Datasets/WHU",
        2: "/vol/research/ak0084_datasets/Datasets/Inria",
        3: "/vol/research/ak0084_datasets/Datasets/GBSS"
    }

    # Selecting the correct dataset based on ds_id
    transfer_input_files = dataset_paths.get(args.ds_id, "")

    # Formatting the file contents
    content = f"""#---------------------------------------------
# Name your batch so it's easy to distinguish in the q.
JobBatchName = {args.JobBatchName}

# --------------------------------------------
# Executable and its arguments
executable    = {args.executable}

arguments     = $ENV(PWD)/main.py --run_mode {args.run_mode} --ds_id {args.ds_id} --model_id {args.model_id} --train_ds_proc_id {args.train_ds_proc_id} --batch_size {args.batch_size} --itr_num {args.iter_num} --MAX_NUM_CROPS {args.MAX_NUM_CROPS} --epochs {args.epochs} --patience {args.patience} --islog2neptune {str(args.islog2neptune)} 

# ---------------------------------------------------
universe         = vanilla

# -------------------------------------------------
# Event, out and error logs
log    = logs/c$(cluster).p$(process).log
output = logs/c$(cluster).p$(process).out
error  = logs/c$(cluster).p$(process).error

# -----------------------------------
# File Transfer, Input, Output
should_transfer_files = YES
transfer_input_files = {transfer_input_files}

# -------------------------------------
# Requirements for the Job
# NOTE: HasStornext is not valid on orca.
#requirements =  (HasStornext) && (CUDACapability > 2.0)

# --------------------------------------
# Resources
request_GPUs     = {args.request_GPUs}
+GPUMem          = {args.GPUMem}

request_CPUs     = {args.request_CPUs}
request_memory   = {args.request_memory}

# This job will complete in less than 1 hour
+JobRunTime = {args.JobRunTime}

# This job can checkpoint
+CanCheckpoint = {args.CanCheckpoint}

# -----------------------------------
# Queue commands
queue {args.queue}
"""

    # Write the content to the .submit_file
    with open(f"{args.files_path}/job_{args.run_mode}_{args.ds_id}_{args.model_id}_{args.train_ds_proc_id}_{args.batch_size}_{args.iter_num}.submit_file", "w") as file:
        file.write(content)

    print(f"Submit file 'files/job_{args.run_mode}_{args.ds_id}_{args.model_id}_{args.train_ds_proc_id}_{args.batch_size}_{args.iter_num}.submit_file' generated successfully.")



