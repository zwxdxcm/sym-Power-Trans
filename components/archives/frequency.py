from scipy.fftpack import dct, idct
import numpy as np


def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def dct_img(img:np.ndarray): 
    if img.ndim != 3: raise NotImplementedError
    transformed = np.zeros(img.shape)
    for i in range(3):  # RGB
        transformed[:, :, i] = dct2(img[:, :, i]) # hwc
    return transformed

def idct_img(transformed):
    recoverd = np.zeros(transformed.shape)
    for i in range(3):  # RGB
        recoverd[:, :, i] = idct2(transformed[:, :, i]) # hwc
    return recoverd


def partition_coff(dct_coff:np.ndarray, num=2):
    # todo: 这个函数可能出了点问题 去网上找一些dct的实现吧 这个系数可能没排列
    total = num * num
    h,w,c = dct_coff.shape
    dct_conf_band = np.zeros((total, h, w, c))
    for j in range(num):
        for i in range(num):
            mask = np.zeros(dct_coff.shape)
            mask[i*int(h/num):(i+1)*int(h/num),:,:] = 1
            mask[:,j*int(w/num):(j+1)*int(w/num),:] = 1
            dct_conf_band[i+j] = np.multiply(mask, dct_coff)
    return dct_conf_band

def reconstruct_coff(dct_conf_band:np.ndarray):
    total = dct_conf_band.shape[0]
    h,w,c = dct_conf_band[0].shape
    dct_coff = np.zeros((h,w,c))
    num = int(np.sqrt(total))
    for j in range(num):
        for i in range(num):
            cur = dct_conf_band[i+j]
            dct_coff[i*int(h/num):(i+1)*int(h/num),:,:] = cur[i*int(h/num):(i+1)*int(h/num),:,:]
            dct_coff[:,j*int(w/num):(j+1)*int(w/num),:] = cur[:,j*int(w/num):(j+1)*int(w/num),:]
    return dct_coff

