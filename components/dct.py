import numpy as np
from scipy.fftpack import dct, idct
import cv2
import matplotlib.pyplot as plt

class DCTImageProcessor:
    def __init__(self):
        self.image = None
        # todo: 设计block_size

    # todo: 可以改成oo的重载 这里    
    def set_img_data(self, data):
        self.image = data

    def dct2(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def pad_image(self, image):
        h, w, c = image.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        return padded_image

    def apply_dct_to_image(self):
        padded_image = self.pad_image(self.image)
        h, w, c = padded_image.shape
        dct_image = np.zeros_like(padded_image, dtype=float)
        
        for ch in range(c):
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    block = padded_image[i:i+8, j:j+8, ch]
                    dct_image[i:i+8, j:j+8, ch] = self.dct2(block)
        
        return dct_image

    def apply_idct_to_image(self, dct_img, mask=None):
        h, w, c = dct_img.shape
        r_image = np.zeros_like(dct_img, dtype=float)
        if mask is None:
            mask = self.gen_filter_mask(0, 8)
        
        for ch in range(c):
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    block = dct_img[i:i+8, j:j+8, ch]
                    filtered_block = block * mask
                    r_image[i:i+8, j:j+8, ch] = self.idct2(filtered_block)
        
        ## with paddings
        r_image = r_image[:self.image.shape[0], :self.image.shape[1], :]
        r_image = np.clip(r_image,0. , 255.) 

        return r_image

    def gen_filter_mask(self, low_freq, high_freq, block_size=8):
        # todo: 目前只支持 low_freq == 0 后续扩展
        if low_freq != 0: raise NotImplementedError
        mask = np.zeros((block_size, block_size), dtype=bool)
        for u in range(block_size):
            for v in range(block_size):
                if low_freq <= u < high_freq and low_freq <= v < high_freq:
                    mask[u, v] = True
        return mask

    def reconstruct_dct_coff(self, base_dct, inject_dct, mask):
        '''base为基础, mask处不替换, 其他地方换成inject dct系数的值'''
        r_dct = np.zeros_like(base_dct, dtype=float)
        h, w, c = base_dct.shape
        for ch in range(c):
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    block = base_dct[i:i+8, j:j+8, ch] * mask + inject_dct[i:i+8, j:j+8, ch] * (~mask)
                    r_dct[i:i+8, j:j+8, ch] = block
        return r_dct
        
    # todo: mask 生成应该放在循环外边 这样循环太多了
    # def filter_frequencies(self, block, low_freq, high_freq):
        # mask = np.zeros_like(block, dtype=bool)
        # for u in range(8):
        #     for v in range(8):
        #         if low_freq <= u < high_freq and low_freq <= v < high_freq:
        #             mask[u, v] = True
        # filtered_block = block * mask
        # return filtered_block


# Example usage
if __name__ == "__main__":
    # processor = DCTImageProcessor('test.png')
    processor = DCTImageProcessor('../data/demo/gate.png')
    
    # Apply DCT to the image
    dct_image = processor.apply_dct_to_image()
    
    # Recover the image from DCT
    recovered_image02 = processor.apply_idct_to_image(low_freq=0, high_freq=2)
    recovered_image04 = processor.apply_idct_to_image(low_freq=0, high_freq=4)
    recovered_image06 = processor.apply_idct_to_image(low_freq=0, high_freq=6)
    recovered_image08 = processor.apply_idct_to_image(low_freq=0, high_freq=8)
    recovered_image24 = processor.apply_idct_to_image(low_freq=2, high_freq=4)
    recovered_image46 = processor.apply_idct_to_image(low_freq=4, high_freq=6)
    recovered_image68 = processor.apply_idct_to_image(low_freq=6, high_freq=8)

    # Display the results
    plt.figure(figsize=(15, 9))
    
    plt.subplot(3, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(processor.image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(3, 3, 2)
    plt.title('DCT Image (log scale)')
    # For visualization, we take the log of the absolute values of the DCT coefficients
    dct_image_log = np.log(np.abs(dct_image) + 1)
    dct_image_log = np.clip(dct_image_log / np.max(dct_image_log), 0, 1)  # Normalize for display
    plt.imshow(dct_image_log)
    
    plt.subplot(3, 3, 3)
    plt.title('Recovered Image 0-2')
    plt.imshow(cv2.cvtColor(recovered_image02.astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    plt.subplot(3, 3, 4)
    plt.title('Recovered Image 0-4')
    plt.imshow(cv2.cvtColor(recovered_image04.astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    plt.subplot(3, 3, 5)
    plt.title('Recovered Image 0-6')
    plt.imshow(cv2.cvtColor(recovered_image06.astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    plt.subplot(3, 3, 6)
    plt.title('Recovered Image 0-8')
    plt.imshow(cv2.cvtColor(recovered_image08.astype(np.uint8), cv2.COLOR_BGR2RGB))

    plt.subplot(3, 3, 7)
    plt.title('Recovered Image 2-4')
    plt.imshow(cv2.cvtColor(recovered_image24.astype(np.uint8), cv2.COLOR_BGR2RGB))

    plt.subplot(3, 3, 8)
    plt.title('Recovered Image 4-6')
    plt.imshow(cv2.cvtColor(recovered_image46.astype(np.uint8), cv2.COLOR_BGR2RGB))

    plt.subplot(3, 3, 9)
    plt.title('Recovered Image 6-8')
    plt.imshow(cv2.cvtColor(recovered_image68.astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.savefig('dct_py_res.jpg')