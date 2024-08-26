import os

from util import misc
from util.tensorboard import writer
from util.logger import log
from opt import opt
from trainer.img_trainer import ImageTrainer
from trainer.video_trainer import VideoTrainer
from trainer.dev_img_trainer import DevImageTrainer
from trainer.audio_trainer import AudioTrainer
import copy
import torch.multiprocessing as mp
import torch.distributed as dist
# from trainer.series_trainer import SeriesTrainer
from multiprocessing import Process, Queue
import torch

# mp.set_start_method('spawn')
# lock = mp.Lock()


def main():
    # use_cuda = 0
    # use_cuda = [1,3,4,5,6,7]
    args = opt()
    misc.fix_seed(args.seed)
    log.inst.success("start")
    writer.init_path(args.save_folder)
    log.set_export(args.save_folder)

    #########################
    #### 训练数据集的case
    if os.path.isdir(args.input_path) and args.signal_type != "video":
        dir = args.input_path
        entries = os.listdir(dir)
        files = [entry for entry in entries if os.path.isfile(os.path.join(dir, entry))]
        files = sorted(files, key= lambda x: int(x.split(".")[0].split("-")[-1])) # todo: adpat to kodak

        # 一张显卡顺序执行
        # @test
        result_list = process_task(files, args, cuda_num=0)
        record_result(result_list, args.save_folder)
        
        # @test
        ### 目前用这个多process的很慢 未解之谜 之后再说 先用一张卡跑所有数据吧
        # task_list = [[] for _ in range(len(use_cuda))]
        
        # # 分配任务
        # for (i,file) in enumerate(files):
        #     to_allocate = i % len(use_cuda)
        #     task_list[to_allocate].append(file)
        

        # # print('task_list: ', task_list)
        
        # processes = []
        # for file_list, cuda_num in zip(task_list, use_cuda):
        #     p = Process(target=process_task, args=(file_list,args, cuda_num))
        #     p.daemon = True
        #     processes.append(p)
        #     p.start()

        # for p in processes:
        #     p.join()
        

    else:
        # 单独训练一个任务 
        result_list = [start_trainer(args)]
        record_result(result_list, args.save_folder)
    
    #########################

    writer.close()
    log.end_all_timer()
    log.inst.success("Done")


def process_task(file_list, args, cuda_num=0):
    # 顺序执行
    # torch.cuda.set_device(cuda_num)
    # torch.set_num_threads(16)
    dir_path = args.input_path
    results = []
    for file in file_list:
        cur_args = copy.deepcopy(args)
        cur_args.device = f"cuda:{cuda_num}"
        cur_args.input_path = os.path.join(dir_path, file)
        cur_res = start_trainer(cur_args)
        results.append(cur_res)
    return results


def start_trainer(args):
    if args.signal_type == "series":
        # trainer = SeriesTrainer(args)
        raise NotImplementedError
    elif args.signal_type == "audio":
        trainer = AudioTrainer(args)
    elif args.signal_type == "image":
        if args.method_type == "dev":
            trainer = DevImageTrainer(args)
        else:
            trainer = ImageTrainer(args)
    elif args.signal_type == "video":
        trainer = VideoTrainer(args)
    trainer.train()
    res = getattr(trainer, "result", None)
    return res


def record_result(result_list: list, path:str):
    
    if None in result_list: return

    keys = result_list[0].keys()
    for key in keys:
        file_path = os.path.join(path, f"res_{key}.txt")
        with open(file_path, 'w') as file:
            for result in result_list:
                psnr_ = result[key]
                file.write(f"{psnr_:.10f}\n")


if __name__ == '__main__':
    main()