import av #pip install av==9.0.1
import sys
import os
from PIL import Image
'''@ songhao'''

def hevc_process(source_path, frames=30):
    hevcs_path = os.listdir(source_path)
    # 所有hevc的路径，存放在一个列表
    hecv_path_list = [(source_path+'/'+hevc) for hevc in hevcs_path if hevc[-5:]=='.hevc']
    print(hevcs_path)
    output_dir = source_path
    i = 0

    #只选一个
    hecv_path_list=["/home/wxzhang/projects/coding4paper/projects/subband/data/uvg/ShakeNDry_3840x2160_120fps_420_8bit_HEVC_RAW.hevc"]
    
    for hecv_file in hecv_path_list:
        container = av.open(hecv_file)
        new_path = os.path.join(output_dir, "hevc_output4", hevcs_path[i][:-6])
        if not(os.path.exists(new_path)):
            os.makedirs(new_path)
        i+=1
        for frame in container.decode(video = 0):
            print("处理前的帧大小: %04d (width: %d, height: %d)" % (frame.index, frame.width, frame.height))
            # 按帧保存图片
            if(frame.index < frames):
                # PYAV库调用了PIL的API，所以可用PIL进行缩放
                image = frame.to_image()
                resized_image = image.resize((960,540))
                resized_image.save("%s%04d.jpg" % (new_path, frame.index+1))
 
if __name__ == "__main__":
    source_path = "/home/wxzhang/projects/coding4paper/projects/subband/data/uvg"
    hevc_process(source_path)
