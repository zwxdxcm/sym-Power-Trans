import os
import glob
import cv2 as cv


input_folder = "/home/ubuntu/projects/coding4paper/projects/subband/log/60.2"
output_path = "/home/wxzhang/projects/coding4paper/projects/subband/script"

input_path_list = glob.glob(os.path.join(input_folder, '*'))
print('input_path_list: ', input_path_list)
output_path = os.path.join(output_path,"visual_img_detail")

os.makedirs(output_path, exist_ok=True)

def crop_and_rec_image(img_path,output_path, key="test"):
    img = cv.imread(img_path) # h,w,c

    # crop image
    left, right, upper,lower = [250,350,550,600]
    cropped_img = img[upper:lower, left:right]
    cv.imwrite(os.path.join(output_path, f"{key}_crop.png"), cropped_img)
    
    # rectangle
    color = (0, 0, 255)  # 红色 (B, G, R)
    thickness = 2  # 框的厚度
    cv.rectangle(img, (left, upper), (right, lower), color, thickness)
    cv.imwrite(os.path.join(output_path, f"{key}_all.png"), img)

# for input_path in input_path_list:
#     key = input_path.split("_")[2]
#     print('key: ', key)
#     target_img = "final_pred_kodim17.png.png"
#     img_path = os.path.join(input_path, target_img)
#     print('img_path: ', img_path)
#     crop_and_rec_image(img_path, output_path, key)


crop_and_rec_image(img_path="/home/wxzhang/projects/coding4paper/projects/subband/data/Kodak/kodim17.png",
                   output_path=output_path,
                   key="gt")