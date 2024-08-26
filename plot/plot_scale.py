import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os


def read_data_from_log(log_path):
    p = Path(log_path)
    subfolders = [x for x in p.iterdir() if x.is_dir()]
    data = {}
    for log in subfolders:
        key = log.name.split("_")[2] # amp
        res_path = os.path.join(str(log), "res_psnr_5000.txt")
        res_list = []
        # read txt
        with open(res_path, 'r') as file:
            for line in file:
                res_list.append(float(line.strip()))
        data[key] = res_list
    
    sorted_data = {k: data[k] for k in sorted(data, key=lambda x: float(x))}
    # del sorted_data['0.9']
    return sorted_data

def plot_scale(data: dict):
    
    x_ = []
    y_ = []
    std_ = [] 
    for key in data.keys():
        x_.append(float(key))
        y_.append(np.mean(data[key]))
        std_.append(np.std(data[key]))
    x_ = np.array(x_)
    y_ = np.array(y_)
    std_ = np.array(std_)
    

    
    plt.figure(figsize=(7, 4))
    plt.rcParams['font.size'] = 16
    
    plt.xticks([0,1, 2, 3,4, 5, 6,7, 8,9,10])
    plt.xlabel('Range('+r'$k$'+')') 
    plt.ylabel('PSNR')
    plt.grid(True,)
    plt.plot(x_,y_,)
    plt.scatter(x_, y_, color='red', marker='D',s=2, zorder=10)
    # plt.ylim(20, 32)
    plt.fill_between(x_, y_ - std_, y_ + std_, color='blue', alpha=0.1, label='Standard Deviation')
    
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2)  
    plt.savefig('./plot_scale.png', bbox_inches='tight') 
    plt.show()

    pass

if __name__ == "__main__":
    # prepare data
    data = read_data_from_log("/home/ubuntu/projects/coding4paper/projects/subband/log/archive/norm_exp_13.2")
    # run
    plot_scale(data)
