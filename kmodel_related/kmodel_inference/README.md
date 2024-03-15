**编译环境**：[k230 sdk docker](https://github.com/kendryte/k230_sdk)；**运行环境**：烧录编译环境对应版本镜像的K230开发板

**注**：

- 大小核共用/sharefs/

- 对于编译上板程序来说，请确保K230_AI_Demo_Development_Process_Analysis放到k230_sdk/src/reference

```bash
cd k230_sdk/src/reference
git clone https://github.com/JayL323/K230_AI_Demo_Development_Process_Analysis.git
```

#debug模式

```bash
#debug模式下生成各种调试时使用可执行文件
./build_app.sh debug
#将生成k230_bin目录拷贝到开发板
#在开发板小核执行
cd /sharefs/
scp -r username@ip:/xxx/k230_bin .
#在开发板大核执行
cd /sharefs/k230_bin/debug
#验证人脸检测上板推理kmodel和simulator推理kmodel相似度
./face_detect_main_nncase.sh    
#验证人脸识别上板推理kmodel和simulator推理kmodel相似度
./face_recognize_main_nncase.sh
```

#release模式

```bash
#release模式下k230_bin生成人脸检测、人脸识别可执行文件
./build_app.sh
#在开发板小核执行
cd /sharefs/
scp -r username@ip:/xxx/k230_bin .
#在开发板大核执行
cd /sharefs/k230_bin/face_detect
#人脸检测推理，按‘q’回车后退出
./face_detect_isp.sh  

cd /sharefs/k230_bin/face_recognize
#人脸识别推理，按‘ESC’退出，按'i'注册，按'r'清空数据库
./face_recognize_isp.sh
```