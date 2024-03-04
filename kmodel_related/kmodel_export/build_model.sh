#!/bin/bash
set -x
mkdir k230_kmodel
k230_kmodel=`pwd`/k230_kmodel
mkdir k230_utils
k230_utils=`pwd`/k230_utils

export NNCASE_PLUGIN_PATH=$NNCASE_PLUGIN_PATH:/usr/local/lib/python3.8/dist-packages/
export PATH=$PATH:/usr/local/lib/python3.8/dist-packages/
source /etc/profile

#build uint8 face detect
cd face_detection
rm -rf bin/output*.bin
rm -rf onnx/*.kmodel
python mobile_retinaface_data_100_640.py --target k230 --model onnx/FaceDetector.onnx --dataset WIDER_val
python mobile_retinaface_onnx_simu_640.py --target k230 --model onnx/FaceDetector.onnx --model_input bin/face_det_0_640x640_float32.bin --kmodel onnx/k230_face_detection_data_100_640.kmodel --kmodel_input bin/face_det_0_640x640_uint8.bin
cp -a onnx/k230_face_detection_data_100_640.kmodel ${k230_kmodel}/face_detect_640.kmodel
cp -a bin/face_det_0_640x640_uint8.bin ${k230_utils}
cp -a bin/face_det_*_k230_simu.bin ${k230_utils}
cp -a bin/face_detect.jpg ${k230_utils}
rm -rf gmodel_dump_dir
rm -rf tmp
cd ..


#build uint8 face recognize
cd face_recognition
rm -rf bin/output*.bin
rm -rf onnx/*.kmodel
python mobile_face.py --target k230 --model onnx/MobileFaceNet.onnx --dataset lfw
python mobile_face_onnx_simu.py --target k230 --model onnx/MobileFaceNet.onnx --model_input bin/face_recg_0_112x112_float32.bin --kmodel onnx/k230_mobile_face.kmodel --kmodel_input bin/face_recg_0_112x112_uint8.bin
cp -a onnx/k230_mobile_face.kmodel ${k230_kmodel}/face_recognize.kmodel
cp -a bin/face_recg_0_112x112_uint8.bin ${k230_utils}
cp -a bin/face_recg_0_k230_simu.bin ${k230_utils}
rm -rf gmodel_dump_dir
rm -rf tmp
cd ..
