import os
from manager import ParamManager
import subprocess
import numpy as np
import math

def run_with_manager():
    '''single task | with tmux''' # 一般用于debug 跑通单例
    pm = ParamManager(model="siren", enum=3)
    pm.p.input_path = "/home/ubuntu/projects/coding4paper/projects/subband/data/text/test_data/01.png"
    pm.p.up_folder_name = "debug"
    pm.p.num_epochs = 2000
    pm.p.log_epoch = 500
    use_cuda = 2
    cmd_str = pm.export_cmd_str(use_cuda=[use_cuda])
    print(f"Running: {cmd_str}")
    os.system(cmd_str)

def run_with_manager_subprocess(model_list, enum_list, gpu_list):
    # subporcess
    
    # model_list = ["siren", "finer"]
    # enum_list = [0,1,2,3,4,5]
    # gpu_list = [0,3,4,5,6,7]

    # model_list = ["siren", "finer"]
    # enum_list = [6,6]
    # gpu_list = [0,3]

    processes = []
    assert len(enum_list) == len(gpu_list)
    for model, enum, use_cuda in zip(model_list, enum_list,gpu_list):
        pm = ParamManager(model= model, enum=enum)
        cmd_str = pm.export_cmd_str(use_cuda=[use_cuda])
        ##  print cmd str for debugger
        # exit()
        process = subprocess.Popen(cmd_str, shell=True)
        print(f"PID: {process.pid}")
        processes.append(process)
    
    for process in processes:
        process.wait()

def debug():
   args =[
       # "--debug", 
       "--up_folder_name", "data_folder",
       "--tag", "normlization",
       "--num_epochs", 
       "5000",
       #"--hidden_features",
       #"350"
        "--lr",
        "5e-3",  # gauss wire 5e-3 | siren finer 5e-4 | pemlp 1e-3
        "--method_type",
        "dev",
        "--model_type",
        "gauss",
        "--normalization",
        "z_power",
        "--pn_beta",
        "0",
        "--pn_buffer",
        "0", # adpative 
        "--pn_k",
        "256",
        # "--pn_alpha",
        # "0.01",
        # "--box_shift",
        # "1.0",
        # "--pn_cum",
        # "0.5",
        # "--amp",
        # "1",
        # "--gamma",
        # "0.353",
    #    "--lambda_l",
    #    "1e-5",
        # "--permute_block",
        # "--dual_low_pass",
        # "4",
        # "--cross_ratio",
        # "0.5",
        # "--lambda_l",
        # "1e-7",
        # "--log_epoch",
        # "1000",
        "--input_path",
        "./data/div2k/test_data/00.png"
        # "./data/div2k/test_data",

   ]
   use_cuda = 6
   os.environ['CUDA_VISIBLE_DEVICES'] = str(use_cuda) # '0,1'
   script = "python main.py " + ' '.join(args)
   print(f"Running: {script}")
   os.system(script)



if __name__ == '__main__':
    # debug()
    # run_with_manager()
    # for model in ["siren", "finer","gauss","wire", "pemlp"]:
    #     run_with_manager_subprocess(model)

    # run_with_manager()
    # exit()

     # 外部循环
    # params:list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [1.2, 1.4, 1.6, 1.8, 2.0] + [3,4,5,6,7]
    # params = [2,3,4,5,6,8,10]
    # params = [1.25,1.5,1.75,2.25,2.75,3.25,3.75,4,25,4.5,4.75] # skewness_a
    params = [0,1,2,3,4,5,6]
    gpu_list = [0,1,3,4,5,6,7]

    params = [0,1,2,3,4,5]
    gpu_list = [0,2,3,4,5,6,7]


    params = [2,3,4,5,6]
    gpu_list = [0,1]

    gpus = len(gpu_list)
    rounds = math.ceil(len(params) / gpus)
    print('rounds: ', rounds)

    for i in range(rounds):
        parmas_list = params[i*gpus: min(len(params),(i+1)*gpus)] # 本质是task list
        cur_len = len(parmas_list)
        model_list =  ["siren" for _ in range(cur_len)]
        gpu_list = gpu_list[:cur_len]
        run_with_manager_subprocess(model_list, parmas_list, gpu_list)
