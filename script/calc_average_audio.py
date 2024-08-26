import os
import glob

# natural test and train

input_folder = "/home/wxzhang/projects/coding4paper/projects/subband/log/902_audio"
input_path_list = glob.glob(os.path.join(input_folder, '*'))
print('input_path_list: ', input_path_list)

for input_path in input_path_list:
    key = input_path.split("_")[3]
    
    print("#"*10, key, "#"*10)
    res = {}
    epochs = [1000, 3000, 5000]
    stats = [ stat for e in epochs for stat in (f"mse_{e}", f"pesq_{e}", f"stoi_{e}", f"sisnr_{e}" )]
    # stats = ["psnr_00", "ssim_500", "psnr_1000", "ssim_1000","psnr_1500"]

    for stats in stats:
        res[stats] = []
        with open(os.path.join(input_path, f"res_{stats}.txt"), 'r') as file:
            for line in file:
                res[stats].append(float(line.strip()))
        _average = sum(res[stats]) / len(res[stats])
        print(f'{stats}_average: ', _average)
