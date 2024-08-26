import os

input_path_list = [
    "/home/ubuntu/projects/coding4paper/projects/subband/log/formal_exps/60.2_finer_finer_2024-07-23-11:17:48",
    "/home/ubuntu/projects/coding4paper/projects/subband/log/formal_exps/60.2_finer_finer-mtrans_2024-07-23-11:17:48",
    "/home/ubuntu/projects/coding4paper/projects/subband/log/formal_exps/60.2_siren_siren_2024-07-23-11:17:48",
    "/home/ubuntu/projects/coding4paper/projects/subband/log/formal_exps/60.2_siren_siren-mtrans_2024-07-23-11:17:48",
    "/home/ubuntu/projects/coding4paper/projects/subband/log/formal_exps/60.2_gauss_gauss_2024-07-23-14:08:17",
    "/home/ubuntu/projects/coding4paper/projects/subband/log/formal_exps/60.2_pemlp_pemlp_2024-07-23-14:08:17",
    "/home/ubuntu/projects/coding4paper/projects/subband/log/formal_exps/60.2_wire_wire_2024-07-23-14:08:17",
]

for input_path in input_path_list:
    key = input_path.split("_")[3]
    
    print("#"*10, key, "#"*10)
    res = {}
    stats = ["psnr_5000", "ssim_5000", "stats_scale", "stats_mean", "stats_std", "stats_skewness"]
    stats = ["psnr_1000", "ssim_1000","psnr_3000", "ssim_3000", "psnr_5000", "ssim_5000"]

    for stats in stats:
        res[stats] = []
        with open(os.path.join(input_path, f"res_{stats}.txt"), 'r') as file:
            for line in file:
                res[stats].append(float(line.strip()))
        _average = sum(res[stats]) / len(res[stats])
        print(f'{stats}_average: ', _average)

        
