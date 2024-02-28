#coding = utf-8
'''使用常用人脸识别训练集主要有：MS1MV2、MS1MV3、Glint360K，
    制作这些数据集一般需要对完整的人脸原图进行【预处理】，
    先进行检测，对每个人脸进行人脸对齐，然后才喂给识别模型'''
import cv2
import argparse
from det import FaceDetectorNet,cfg_mnet
import numpy as np
from skimage import transform as trans

def st_image(ori_image, landmarks):
    #标准正脸人脸五官未知
    le_g = [38.2946, 51.6963]
    re_g = [73.5318, 51.5014]
    nose_g = [56.0252, 71.7366]
    l_mouth_g = [41.5493, 92.3655]
    r_mouth_g = [70.7299, 92.2041]
    #实际人脸五官位置
    le = landmarks[0, :]
    re = landmarks[1, :]
    nose = landmarks[2, :]
    l_mouth = landmarks[3, :]
    r_mouth = landmarks[4, :]
    landmark_get = np.float32([le, re, nose, l_mouth, r_mouth])
    landmark_golden = np.float32([le_g, re_g, nose_g, l_mouth_g, r_mouth_g])
    #计算从实际人脸->标准正脸需要经过的变换
    tform = trans.SimilarityTransform()
    tform.estimate(np.array(landmark_get), landmark_golden)
    M = tform.params[0:2, :]
    #得到变换后的人脸
    affine_output = cv2.warpAffine(ori_image, M, (112, 112), borderValue=0.0)
    return affine_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('--onnx', default='./onnx/FaceDetector.onnx',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--confidence_threshold', default=0.6, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()

    face_det = FaceDetectorNet(cfg_mnet,args)

    img_lists = ['bin/1.png','bin/2.png','bin/3.png']
    for img_file in img_lists:
        ori_img = cv2.imread(img_file)
        results = face_det.run(ori_img)
        for ret in results:
            landms = ret[5:15].reshape((5,2))
            affine_out = st_image(ori_img, landms)
            name = img_file.replace('.','_affine.')
            cv2.imwrite(name, affine_out)

        ori_img = face_det.draw_result(ori_img, results)

        cv2.imshow('retina face detect', ori_img)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()
