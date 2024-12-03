import subprocess
import os
from single_job import generate_submit_file

class JobArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if not os.path.exists("files"):
    os.makedirs("files")
    print("Directory 'files' created.")

run_modes_names = ["train", "test", "continue training"]
ds_names = ["Massa", "WHU", "Inria", "GBSS"]
models_names = ["FPN", "Unet", "UNet++", "DeepLabV3+", "PSPNet", "5 NO", "6 NO", "7 NO", "8 NO", "9 NO", "BSNet"]
ds_preprocessing = ["random cropping", "selected cropping"]

run_specific_job = False

if not run_specific_job:
    run_mode = 0
    batch_size = 6
    if run_mode == 0: # train
        GPU_MEMs = [6000, 9000, 15000, 9000]
        request_memory = ["10G", "10G", "40G", "20G"]
    else: # test
        GPU_MEMs = [7000, 9000, 15000, 9000]
        request_memory = ["5G", "10G", "40G", "10G"]

    for ds_id in range(0, 1): #0 to 5
        for model_id in range(0, 1): #0 to 5
            for train_ds_proc_id in range(1, 2): #0 to 2
                for iter_num in range(0, 5): #0 to 5
                    args = JobArgs(
                        JobBatchName=f"Expr {run_modes_names[run_mode]} - {ds_names[ds_id]} - {models_names[model_id]} - {ds_preprocessing[train_ds_proc_id]}"
                                     f"- BSz {batch_size} - GPUMem {GPU_MEMs[ds_id]} - memory {request_memory[ds_id]} - itr {iter_num}",
                        files_path="files",
                        executable="/user/HS301/ak0084/miniconda3/envs/xML_Alpha/bin/python",
                        run_mode=run_mode, ds_id=ds_id, model_id=model_id, train_ds_proc_id=train_ds_proc_id,
                        batch_size=batch_size, iter_num=iter_num, MAX_NUM_CROPS=50, epochs=200, patience=20, islog2neptune=True,
                        request_GPUs=1, GPUMem=GPU_MEMs[ds_id], request_CPUs=4, request_memory=request_memory[ds_id],
                        JobRunTime=20, CanCheckpoint="false", queue=1)

                    generate_submit_file(args)
                    try:
                        submit_file_name = f"files/job_{run_mode}_{ds_id}_{model_id}_{train_ds_proc_id}_{batch_size}_{iter_num}.submit_file"
                        subprocess.run(["condor_submit", submit_file_name], check=True)
                        print("Job submitted successfully.")
                    except subprocess.CalledProcessError as e:
                        print(f"Job submission failed: {e}")

else:
    run_mode = 0
    batch_size = 32
    ds_id = 2
    model_id = 10
    train_ds_proc_id = 1
    GPU_MEM = 15000
    request_memory = "40G"
    iter_num = 0
    args = JobArgs(
        JobBatchName=f"Expr {run_modes_names[run_mode]} - {ds_names[ds_id]} - {models_names[model_id]} - {ds_preprocessing[train_ds_proc_id]}"
                                 f"- BSz {batch_size} - GPUMem {GPU_MEM} - memory {request_memory} - itr {iter_num}",
        files_path="files",
        executable="/user/HS301/ak0084/miniconda3/envs/xML_Alpha/bin/python",
        run_mode=run_mode, ds_id=ds_id, model_id=model_id, train_ds_proc_id=train_ds_proc_id,
        batch_size=batch_size, iter_num=iter_num, MAX_NUM_CROPS=50, epochs=200, patience=20, islog2neptune=True,
        request_GPUs=1, GPUMem=GPU_MEM, request_CPUs=4, request_memory=request_memory, JobRunTime=20,
        CanCheckpoint="false", queue=1)

    generate_submit_file(args)

    try:
        submit_file_name = f"files/job_{run_mode}_{ds_id}_{model_id}_{train_ds_proc_id}_{batch_size}_{iter_num}.submit_file"
        subprocess.run(["condor_submit", submit_file_name], check=True)
        print("Job submitted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Job submission failed: {e}")