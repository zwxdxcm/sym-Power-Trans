import os
import glob

# natural test and train

input_folder = "/home/ubuntu/projects/coding4paper/projects/subband/log/60.31"
input_path_list = glob.glob(os.path.join(input_folder, '*'))
print('input_path_list: ', input_path_list)

for input_path in input_path_list:
    key = input_path.split("_")[2]
    
    print("#"*10, key, "#"*10)
    res = {}
    epochs = [100, 300, 500]
    stats = [ stat for e in epochs for stat in (f"psnr_{e}", f"ssim_{e}" )]
    # stats = ["psnr_00", "ssim_500", "psnr_1000", "ssim_1000","psnr_1500"]

    for stats in stats:
        res[stats] = []
        with open(os.path.join(input_path, f"res_{stats}.txt"), 'r') as file:
            for line in file:
                res[stats].append(float(line.strip()))
        _average = sum(res[stats]) / len(res[stats])
        print(f'{stats}_average: ', _average)
