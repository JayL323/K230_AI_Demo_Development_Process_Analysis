**目录说明**：

```bash
.
├── kmodel_related       #kmodel相关              
│   ├── kmodel_export    #kmodel导出
│   └── kmodel_inference #kmodel推理
└── onnx_related         #onnx相关
    ├── onnx_export      #onnx导出
    └── onnx_inference   #onnx推理
```

**使用顺序**：

1. `onnx_related/onnx_export`：onnx导出；**解释、运行环境**：普通python环境
2. `onnx_related/onnx_inference`：onnx推理；**解释、运行环境**：普通python环境
3. `kmodel_related/kmodel_export`：kmodel导出；**解释、运行环境**：建议[k230 sdk docker](https://github.com/kendryte/k230_sdk)，安装nncase
4. `kmodel_related/kmodel_inference`：kmodel推理；**编译环境**：[k230 sdk docker](https://github.com/kendryte/k230_sdk)；**运行环境**：烧录编译环境对应版本镜像的K230开发板

**详细使用流程**：（对应链接）