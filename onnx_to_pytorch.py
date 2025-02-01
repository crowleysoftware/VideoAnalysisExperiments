import onnx
from onnx2pytorch import ConvertModel


onnx_model = onnx.load('"C:/Users/Administrator/Downloads/discvisioon.ONNX/model.onnx"')
pytorch_model = ConvertModel(onnx_model)
