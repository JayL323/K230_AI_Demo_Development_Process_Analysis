**解释、运行环境**：普通python环境

```bash
#将人脸检测模型转换为ONNX
1.下载人脸检测repo
git clone https://github.com/biubug6/Pytorch_Retinaface
2.将face_detection_convert_to_onnx.py拷贝到Pytorch_Retinaface中
3.根据Pytorch_Retinaface说明文档下载预训练模型
4.生成ONNX模型
python face_detection_convert_to_onnx.py

#将人脸识别模型转换为ONNX
1.下载人脸识别repo
git clone https://github.com/Xiaoccer/MobileFaceNet_Pytorch
2.将face_recognition_convert_to_onnx.py拷贝到MobileFaceNet_Pytorch中
3.生成ONNX模型
python face_recognition_convert_to_onnx.py
```

