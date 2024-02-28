# coding = utf-8
import os
import numpy as np
from det import OnnxModel

class MobileFaceNet():
    def __init__(self, onnx_path=None):
        if onnx_path is None:
            root_pth = os.path.dirname(os.path.abspath(__file__))
            dat_path = os.path.join(root_pth, 'onnx/MobileFaceNet.onnx')
        self.normalize_mean = 127.5
        self.normalize_std = 128.0
        self.model = OnnxModel(dat_path)

    def pre_process(self,img,to_bin = True):
        img = img[..., ::-1]

        if to_bin:
            model_name = 'face_recg'
            input_height,input_width = 112,112
            img_copy = img.copy()
            img_copy = np.transpose(img_copy, [2, 0, 1])
            kmodel_input_bin_file = 'bin/{}_{}_{}x{}_uint8.bin'.format(model_name, 0, input_height, input_width)
            img_copy.tofile(kmodel_input_bin_file)

        img = np.array(img, dtype='float32')
        # 归一化
        for i in range(3):
            img[:, :, i] -= self.normalize_mean
            img[:, :, i] /= self.normalize_std
        img = np.transpose(img, [2, 0, 1])

        if to_bin:
            model_name = 'face_recg'
            input_height, input_width = 112, 112
            onnx_input_bin_file = 'bin/{}_{}_{}x{}_float32.bin'.format(model_name, 0, input_height, input_width)
            img.tofile(onnx_input_bin_file)

        input_data = np.expand_dims(img, 0)
        return input_data

    def farward(self, input_data):
        embedding = self.model.forward(input_data)
        return embedding[0]

