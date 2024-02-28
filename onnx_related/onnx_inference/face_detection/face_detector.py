# coding = utf-8
import os
import cv2
import numpy as np
import argparse
from onnx_model import OnnxModel
from data_augment import pad_to_square,resize_subtract_mean
from prior_box import PriorBox
from box_utils import decode,decode_landm
from py_cpu_nms import py_cpu_nms
from config import cfg_mnet

#face_detector.py
class FaceDetectorNet():
    def __init__(self, cfg, args):
        self.normalize_mean = [104, 117, 123]
        self.normalize_std = [1, 1, 1]
        self.model = OnnxModel(args.onnx)
        self.in_size = self.model.input_tensors[0].shape[2]

        self.cfg = cfg
        self.model_height, self.model_width = self.model.input_tensors[0].shape[2], self.model.input_tensors[0].shape[3]  # onnx input shape
        priorbox = PriorBox(self.cfg, image_size=(self.model_height, self.model_width))
        priors = priorbox.forward()
        self.priors_numpy = priors.numpy()

    def pre_process(self,ori_img):
        max_ori_img = max(ori_img.shape[1], ori_img.shape[0])
        self.scale = [max_ori_img] * 4
        self.scale1 = [max_ori_img] * 10
        pad_img = pad_to_square(ori_img,self.normalize_mean,True)
        resize_img = resize_subtract_mean(pad_img,self.in_size,self.normalize_mean)
        resize_img_float = np.float32(resize_img)
        input_data = np.expand_dims(resize_img_float, 0)
        return input_data

    def post_process(self,loc,conf,landms):
        loc, conf, landms = loc[0],conf[0],landms[0]
        boxes = decode(loc, self.priors_numpy, self.cfg['variance'])
        boxes = boxes * self.scale / 1                     #右、下padding
        scores = conf[:, 1]
        landms = decode_landm(landms, self.priors_numpy, self.cfg['variance'])

        landms = landms * self.scale1 / 1                  #右、下padding

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        results = np.concatenate((dets, landms), axis=1)
        return results

    def draw_result(self,img_raw,results):
        # show image
        for b in results:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

        if args.save_image:
            name = "bin/test_out.jpg"
            cv2.imwrite(name, img_raw)
        return img_raw

    def run(self, img_raw):
        input_data = self.pre_process(img_raw)
        loc,conf,landms = self.model.forward(input_data)
        results = self.post_process(loc,conf,landms)
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('--onnx', default='./onnx/FaceDetector.onnx',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()

    face_det = FaceDetectorNet(cfg_mnet,args)

    ori_img = cv2.imread('bin/test.jpg')
    # ori_img = cv2.resize(ori_img,(640,640))
    results = face_det.run(ori_img)
    ori_img = face_det.draw_result(ori_img, results)

    cv2.imshow('retina face detect', ori_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

