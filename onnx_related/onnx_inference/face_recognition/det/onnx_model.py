# coding = utf-8
import onnxruntime as ort

class OnnxModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.sess = ort.InferenceSession(onnx_path)
        self.in_names = [input.name for input in self.sess.get_inputs()]
        self.out_names = [output.name for output in self.sess.get_outputs()]
        self.input_tensors = self.sess.get_inputs()
        self.output_tensors = self.sess.get_outputs()

    def get_input_feed(self, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        if type(image_tensor) is not list:
            image_tensor = [image_tensor]
        for in_id in range(len(self.in_names)):
            input_feed[self.in_names[in_id]] = image_tensor[in_id]
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        input_feed = self.get_input_feed(image_tensor)
        output = self.sess.run(self.out_names, input_feed=input_feed)
        return output

if __name__ == '__main__':
    onnx_pth = './onnx/FaceDetector.onnx'
    onnx_model = OnnxModel(onnx_pth)
    print('in shape:')
    for input_tensor in onnx_model.input_tensors:
        print(input_tensor.shape)

    print('out shape:')
    for output_tensor in onnx_model.output_tensors:
        print(output_tensor.shape)
