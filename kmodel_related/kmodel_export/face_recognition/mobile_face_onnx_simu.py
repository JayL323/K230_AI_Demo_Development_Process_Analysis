import os
import copy
import argparse
import numpy as np
import onnx
import onnxruntime as ort
import nncase

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

def cosine(gt, pred):
    return (gt @ pred) / (np.linalg.norm(gt, 2) * np.linalg.norm(pred, 2))

def main():
    parser = argparse.ArgumentParser(prog="nncase")
    parser.add_argument("--target", type=str, help='target to run')
    parser.add_argument("--model", type=str, help='original model file')
    parser.add_argument("--model_input", type=str, help='input bin file for original model')
    parser.add_argument("--kmodel", type=str, help='kmodel file')
    parser.add_argument("--kmodel_input", type=str, help='input bin file for kmodel')
    args = parser.parse_args()

    # 1. onnx推理，得到cpu_results
    ort_session = ort.InferenceSession(args.model)
    output_names = []
    model_outputs = ort_session.get_outputs()
    for i in range(len(model_outputs)):
        output_names.append(model_outputs[i].name)
    model_input = ort_session.get_inputs()[0]
  
    model_input_name = model_input.name
    model_input_type = np.float32
    model_input_shape = model_input.shape
    print('onnx_input：',model_input_shape)
    model_input_data = np.fromfile(args.model_input, model_input_type).reshape(model_input_shape)
    cpu_results = []
    cpu_results = ort_session.run(output_names, { model_input_name : model_input_data })

    
    # 2. simulator推理，得到nncase_results
    # create simulator
    sim = nncase.Simulator()

    # read kmodel
    kmodel = read_model_file(args.kmodel)

    # load kmodel
    sim.load_model(kmodel)

    # read input.bin
    input_shape = [1, 3, 112, 112]
    dtype = sim.get_input_desc(0).dtype
    input = np.fromfile(args.kmodel_input, dtype).reshape(input_shape)

    # set input for simulator
    sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input))

    # simulator inference
    nncase_results = []
    sim.run()
    for i in range(sim.outputs_size):
        nncase_result = sim.get_output_tensor(i).to_numpy()
        # print("nncase_result:",nncase_result)
        input_bin_file = 'bin/face_recg_{}_{}_simu.bin'.format(i,args.target)
        nncase_result.tofile(input_bin_file)
        nncase_results.append(copy.deepcopy(nncase_result))

    # 3. 计算onnx和simulator相似度
    for i in range(sim.outputs_size):
        cos = cosine(np.reshape(nncase_results[i], (-1)), np.reshape(cpu_results[i], (-1)))
        print('output {0} cosine similarity : {1}'.format(i, cos))

if __name__ == '__main__':
    main()