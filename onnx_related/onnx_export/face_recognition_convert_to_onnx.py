import numpy as np
import torch
from core import model


def convert_onnx(net, path_module, output, opset=11):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    ckpt = torch.load(path_module,map_location='cpu')
    net.load_state_dict(ckpt['net_state_dict'],strict=True)
    net.eval()
    torch.onnx.export(net, img, output, input_names=["data"], keep_initializers_as_inputs=False, verbose=False,
                      opset_version=opset)
    # model = onnx.load(output)
    # graph = model.graph
    # graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    # if simplify:
    #     from onnxsim import simplify
    #     model, check = simplify(model)
    #     assert check, "Simplified ONNX model could not be validated"
    # onnx.save(model, output)


if __name__ == '__main__':
    net = model.MobileFacenet()
    input_file = 'model/best/068.ckpt'
    output_file = 'model/best/MobileFaceNet.onnx'
    convert_onnx(net, input_file, output_file)