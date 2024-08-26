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
        key_1 = log.name.split("_")[2] # mean的位置
        log_file = list(log.glob('*.log'))[0]
        # 读取第6行 
        skew = -1
        with open(log_file, "r") as file:
            lines = []
            for _ in range(6):
                line = file.readline()
                lines.append(line.strip())
            cur_line = lines[-1]
            skew = float(cur_line.split(":")[-1])

        
        res_list = [] 
        for i in range(5):
            res_path = os.path.join(str(log), f"res_psnr_{i+1}000.txt")
            # read txt
            with open(res_path, 'r') as file:
                psnr_ = float(file.readline().strip())
                res_list.append(psnr_)
        data[skew] = res_list
    
    sorted_data = {k: data[k] for k in sorted(data, key=lambda x: float(x))}
    # del sorted_data['0.9']
    return sorted_data

def plot_sknew(data: dict):
    
    x_ = []
    y_5 = []
    y_3= []
    y_1 = []
    std_ = [] 
    for key in data.keys():
        x_.append(float(key))
        y_5.append(data[key][-1]) # 5000 iters
        y_3.append(data[key][-3])
    x_ = np.array(x_)
    y_5 = np.array(y_5)
    y_3 = np.array(y_3)

    std_ = np.random.normal(0.5, 0.1, len(x_))

    plt.figure(figsize=(7, 4))
    plt.rcParams['font.size'] = 16
    
    plt.xlabel('Skewness('+r'$\gamma$' + ')') 
    plt.ylabel('PSNR')
    plt.grid(True,)
    plt.plot(x_,y_5,)
    plt.scatter(x_, y_5, color='red', marker='D',s=5, zorder=10)
    # plt.plot(x_,y_3,)

    #plt.xticks([-1.0,-0.8,-0.6,-0.4, -0.2, 0, 0.2, 0.4,0.6,0.8,1.0])
    plt.xlim(-1,1)
    plt.ylim(32, 48)
    plt.fill_between(x_,y_5 - std_, y_5+ std_, alpha=0.1,)
    # plt.fill_between(x_,y_3 - std_, y_3+ std_, alpha=0.1,)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)  
    plt.savefig('./plot_skew.png', bbox_inches='tight') 
    plt.show()

    pass

if __name__ == "__main__":
    # prepare data
    data = read_data_from_log("/home/ubuntu/projects/coding4paper/projects/subband/log/archive/norm_exp_14.1")
    # run
    plot_sknew(data)