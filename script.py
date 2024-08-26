from util import misc
import os
from components.img_analyser import ImgAnalyser
from tqdm import tqdm
from loguru import logger


def run_script():
    cur_time = misc.gen_cur_time()
    log_path = os.path.join("./log", "analysis", f"{cur_time}")
    os.makedirs(log_path, exist_ok=True)

    _set_logger(log_path)

    input_path = "./data/div2k/test_data"
    entries = os.listdir(input_path)
    files = [
        entry for entry in entries if os.path.isfile(os.path.join(input_path, entry))
    ]
    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    # demo
    # files = [files[2]]
    
    results = []

    for img in tqdm(files):
        cur_input_path = os.path.join(input_path, img)
        img_name= os.path.splitext(img)[0]
        analyser = ImgAnalyser(cur_input_path, log_path)
        
        analyser.visual_histc(
            analyser.min_max_img, f"{img_name}_min_hist"
        )
        analyser.visual_histc(
            analyser.bn_img, f"{img_name}_bn_hist"
        )
        analyser.visual_histc(
            analyser.power_img, f"{img_name}_gamma_{analyser.gamma}_hist"
        )

        analyser.profile_penalty(analyser.min_max_img,"minmax")
        analyser.profile_penalty(analyser.power_img,"power")
        analyser.get_penalty_gamma()

        # analyser.visual_histc(
        #     analyser.box_img, f"{img_name}_box_hist"
        # )
        # analyser.visual_histc(
        #     analyser.power_mle_img, f"{img_name}_mle_gamma_{analyser.result['mle_gamma']}_hist"
        # )

        #  analyser.visual_histc(analyser.bn_img[0], f"{img_name}_bn_hist")
        # equ = analyser.opencv_hist(analyser.input_img[0])
        # analyser.get_pdf(analyser.input_img)
        # analyser.visual_histc(analyser.min_max_img, f"{img_name}_hist")
        # analyser.visual_histc(equ, f"{img_name}_equ_hist")

        # dump result
        results.append(analyser.result)
    
    record_result(results, log_path)
                
def record_result(result_list: list, path:str):
    for key in result_list[0].keys():
        file_path = os.path.join(path, f"res_{key}.txt")
        with open(file_path, 'w') as file:
            for result in result_list:
                psnr_ = result[key]
                file.write(f"{psnr_:.4f}\n")

def _set_logger(log_path, name="analysis"):
    path_ = os.path.join(log_path, f"{name}.log")
    logger.add(
            path_,
            level="TRACE",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level.icon} | {message}",
            colorize=True,
        )

if __name__ == "__main__":
    misc.fix_seed(3047)
    run_script()
