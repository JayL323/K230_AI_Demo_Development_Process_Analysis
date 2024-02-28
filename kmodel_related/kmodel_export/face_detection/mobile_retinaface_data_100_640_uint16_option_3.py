import os
import argparse
import numpy as np
from PIL import Image
import onnxsim
import onnx
import nncase

def parse_model_input_output(model_file):
    onnx_model = onnx.load(model_file)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))
    input_tensors = [
        node for node in onnx_model.graph.input if node.name in input_names]

    # input
    inputs = []
    for _, e in enumerate(input_tensors):
        onnx_type = e.type.tensor_type
        input_dict = {}
        input_dict['name'] = e.name
        input_dict['dtype'] = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type.elem_type]
        input_dict['shape'] = [(i.dim_value if i.dim_value != 0 else d) for i, d in zip(
            onnx_type.shape.dim, [1, 3, 640, 640])]
        inputs.append(input_dict)

    return onnx_model, inputs


def onnx_simplify(model_file, dump_dir):
    onnx_model, inputs = parse_model_input_output(model_file)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    input_shapes = {}
    for input in inputs:
        input_shapes[input['name']] = input['shape']

    onnx_model, check = onnxsim.simplify(onnx_model, input_shapes=input_shapes)
    assert check, "Simplified ONNX model could not be validated"

    model_file = os.path.join(dump_dir, 'simplified.onnx')
    onnx.save_model(onnx_model, model_file)
    return model_file


def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

def generate_data_ramdom(shape, batch):
    data = []
    for i in range(batch):
        data.append([np.random.randint(0, 256, shape).astype(np.uint8)])
    return data


def generate_data(shape, batch, calib_dir):
    img_paths = [os.path.join(calib_dir, p) for p in os.listdir(calib_dir)]
    data = []
    for i in range(batch):
        assert i < len(img_paths), "calibration images not enough."
        img_data = Image.open(img_paths[i]).convert('RGB')
        img_data = img_data.resize((shape[3], shape[2]), Image.BILINEAR)
        img_data = np.asarray(img_data, dtype=np.uint8)
        img_data = np.transpose(img_data, (2, 0, 1))
        data.append([img_data[np.newaxis, ...]])
    return np.array(data)

def main():
    parser = argparse.ArgumentParser(prog="nncase")
    parser.add_argument("--target",type=str, help='target to run')
    parser.add_argument("--model",type=str, help='model file')
    parser.add_argument("--dataset", type=str, help='calibration_dataset')

    args = parser.parse_args()

    #临时文件目录
    dump_dir = 'tmp/mobile_retinaface'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    ############################设置参数#########################
    # 1. 设置编译参数，compile_options
    compile_options = nncase.CompileOptions()
    # 指定编译目标, 如'cpu', 'k230',cpu生成cpu上推理的kmodel,k230生成在k230(kpu)上推理的kmodel
    compile_options.target = args.target
    # 预处理
    compile_options.preprocess = True
    # （1）预处理---Transpose相关参数
    # 输入数据的shape，默认为[]。当 preprocess为 True时，必须指定
    input_shape = [1, 3, 640, 640]
    compile_options.input_shape = input_shape
    # 输入数据的layout，默认为""
    # compile_options.input_layout = "NCHW"
    compile_options.input_layout = "0,1,2,3"
    
    # （2）预处理---SwapRB相关参数
    compile_options.swapRB = True
    
    # （3）预处理---Dequantize（反量化）相关参数
    # 开启预处理时指定输入数据类型，默认为"float"；当 preprocess为 True时，必须指定为"uint8"或者"float32"
    compile_options.input_type = 'uint8'            
    # input_type=‘uint8’时反量化有效，反量化之后的数据范围
    compile_options.input_range = [0, 255]

    # （4）预处理---Normalization相关参数
    compile_options.mean = [ 104,117,123]
    compile_options.std = [1, 1, 1]

    # 后处理
    # compile_options.output_layout = "NCHW"
    #Compiler类, 根据编译参数配置Compiler，用于编译神经网络模型
    compiler = nncase.Compiler(compile_options)
    
    # 2. 设置导入参数，import_options（一般默认即可）
    import_options = nncase.ImportOptions()
    model_file = onnx_simplify(args.model, dump_dir)
    model_content = read_model_file(model_file)
    compiler.import_onnx(model_content, import_options)
    
    # 3. 设置量化参数，ptq_options
    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = 100                #量化数量一般是100即可，根据实际情况调整
    ptq_options.set_tensor_data(generate_data(input_shape, ptq_options.samples_count, args.dataset))
    # 四种option可以根据需要选用，暂不支持w_quant_type、quant_type【同时】int16量化
    # option 1：使用'Noclip'的int16量化，权重：float32->int16
    # ptq_options.calibrate_method = 'NoClip'       #量化方法，及变换方法，比较典型的是最大最小值量化
    # ptq_options.w_quant_type = 'int16'            #指定权重（Weights）量化类型，'int16'或'uint8'
    
    # option 2：使用'Noclip'的int16量化，数据：float32->int16
    # ptq_options.calibrate_method = 'NoClip'
    # ptq_options.quant_type = 'int16'               #指定数据（data）量化类型，'int16'或'uint8'
    
    # option 3：使用'Kld'的int16量化，权重：float32->int16
    ptq_options.w_quant_type = 'int16'
    
    # option 4：使用'Kld'的int16量化，数据：float32->int16
    ptq_options.quant_type = 'int16'
    compiler.use_ptq(ptq_options)
    ############################设置参数#########################

    # 4.编译神经网络模型（kmodel）
    compiler.compile()

    # 5.保存kmodel
    kmodel = compiler.gencode_tobytes()
    with open('./onnx/{}_face_detection_data_100_640_int16_option_3.kmodel'.format(args.target), 'wb') as f:
        f.write(kmodel)

if __name__ == '__main__':
    main()