import onnx_graphsurgeon as og
import onnx
import numpy as np
graph =og.import_onnx(onnx.load('yolov8s-dy.onnx'))
graph.inputs[0].name = 'input'
onnx.save(og.export_onnx(graph), 'yolov8s-dy-input.onnx')


