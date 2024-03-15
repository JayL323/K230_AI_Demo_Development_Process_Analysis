#!/bin/bash
set -x
mkdir k230_kmodel
k230_kmodel=`pwd`/k230_kmodel
mkdir k230_utils
k230_utils=`pwd`/k230_utils

export NNCASE_PLUGIN_PATH=$NNCASE_PLUGIN_PATH:/usr/local/lib/python3.8/dist-packages/
export PATH=$PATH:/usr/local/lib/python3.8/dist-packages/
source /etc/profile

#build int16 face detect
#option 1
cd face_detection
rm -rf bin/output*.bin
rm -rf onnx/*.kmodel
python mobile_retinaface_data_100_640_uint16_option_1.py --target k230 --model onnx/FaceDetector.onnx --dataset WIDER_val
python mobile_retinaface_onnx_simu_640.py --target k230 --model onnx/FaceDetector.onnx --model_input bin/face_det_0_640x640_float32.bin --kmodel onnx/k230_face_detection_data_100_640_int16_option_1.kmodel --kmodel_input bin/face_det_0_640x640_uint8.bin
cp -a onnx/k230_face_detection_data_100_640_int16_option_1.kmodel ${k230_kmodel}/face_detect_int16_opt1.kmodel
rm -rf gmodel_dump_dir
rm -rf tmp
cd ..


#option 2
cd face_detection
rm -rf bin/output*.bin
rm -rf onnx/*.kmodel
python mobile_retinaface_data_100_640_uint16_option_2.py --target k230 --model onnx/FaceDetector.onnx --dataset WIDER_val
python mobile_retinaface_onnx_simu_640.py --target k230 --model onnx/FaceDetector.onnx --model_input bin/face_det_0_640x640_float32.bin --kmodel onnx/k230_face_detection_data_100_640_int16_option_2.kmodel --kmodel_input bin/face_det_0_640x640_uint8.bin
cp -a onnx/k230_face_detection_data_100_640_int16_option_2.kmodel ${k230_kmodel}/face_detect_int16_opt2.kmodel
rm -rf gmodel_dump_dir
rm -rf tmp
cd ..

#option 3
cd face_detection
rm -rf bin/output*.bin
rm -rf onnx/*.kmodel
python mobile_retinaface_data_100_640_uint16_option_3.py --target k230 --model onnx/FaceDetector.onnx --dataset WIDER_val
python mobile_retinaface_onnx_simu_640.py --target k230 --model onnx/FaceDetector.onnx --model_input bin/face_det_0_640x640_float32.bin --kmodel onnx/k230_face_detection_data_100_640_int16_option_3.kmodel --kmodel_input bin/face_det_0_640x640_uint8.bin
cp -a onnx/k230_face_detection_data_100_640_int16_option_3.kmodel ${k230_kmodel}/face_detect_int16_opt3.kmodel
rm -rf gmodel_dump_dir
rm -rf tmp
cd ..

#option 4
cd face_detection
rm -rf bin/output*.bin
rm -rf onnx/*.kmodel
python mobile_retinaface_data_100_640_uint16_option_4.py --target k230 --model onnx/FaceDetector.onnx --dataset WIDER_val
python mobile_retinaface_onnx_simu_640.py --target k230 --model onnx/FaceDetector.onnx --model_input bin/face_det_0_640x640_float32.bin --kmodel onnx/k230_face_detection_data_100_640_int16_option_4.kmodel --kmodel_input bin/face_det_0_640x640_uint8.bin
cp -a onnx/k230_face_detection_data_100_640_int16_option_4.kmodel ${k230_kmodel}/face_detect_int16_opt4.kmodel
rm -rf gmodel_dump_dir
rm -rf tmp
cd ..
