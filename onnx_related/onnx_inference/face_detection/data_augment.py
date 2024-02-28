import numpy as np
import cv2
import random
import os

def pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t

def resize_subtract_mean(image, insize, rgb_mean,to_bin=True):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    # interp_method = interp_methods[random.randrange(5)]
    interp_method = interp_methods[0]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    if to_bin:
        # （1）kmodel input to bin
        image_copy = image.copy()
        image_copy = image_copy[:,:,::-1]
        input_bin = image_copy.transpose(2, 0, 1)
        # {任务名称}_{输入index}_{kmodel分辨率宽}_{kmodel分辨率高}_{kmodel输入类型}
        kmodel_in_file = 'bin/{}_{}_{}x{}_uint8.bin'.format('face_det', 0, 640, 640)
        input_bin.tofile(kmodel_in_file)

    image = image.astype(np.float32)
    image -= rgb_mean
    image = image.transpose(2, 0, 1)

    if to_bin:
        # （2）onnx intput to bin
        # {任务名称}_{输入index}_{onnx分辨率宽}_{onnx分辨率高}_{onnx输入类型}
        onnx_in_file = 'bin/{}_{}_{}x{}_float32.bin'.format('face_det', 0, 640, 640)
        image.tofile(onnx_in_file)

    return image