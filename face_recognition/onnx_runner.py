import cv2
import numpy as np
import onnxruntime as ort


class ONNXModelONNXRuntime():
    def __init__(self, path):
        self.session = ort.InferenceSession(path)
        assert len(self.session.get_inputs()) == 1  # support only one input argument
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, input):
        if input.dtype == np.float32:
            pass
        elif input.dtype == np.float64:
            input = input.astype(np.float32)
        else:
            raise ValueError(f"Unexpected input type {input.dtype}")
        
        return self.session.run(None, {self.input_name: input})


def load_model(path, cls=None):
    cls =  ONNXModelONNXRuntime
    return cls(path)
