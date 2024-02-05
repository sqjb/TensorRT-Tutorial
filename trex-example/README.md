![Alt text](image.png)

### installation
1. install tensorrt && yolov8
2. install TensorRT/tools/export
```shell
# step-1: clone the tensorrt repo.
$ git clone https://github.com/NVIDIA/TensorRT.git

# step-2: install the trex tool
$ cd TensorRT/tools/experimental/trt-engine-explorer
$ python3 -m pip install -e .
$ sudo apt-get --yes install graphviz
```

### draw computational graph
1. generate yolov8 onnx
```bash
$ yolo export model=yolov8s.pt format=onnx dynamic=true
```
2. generate TRT engine graph by trex tool
```bash
$ cd trex-example
$ mkdir output
# run the script and check svg in output folder
$ python ../TensorRT/tools/experimental/trt-engine-explorer/utils/process_engine.py \
        ../yolov8-tensorrt/yolov8s.onnx \
        output \
        shapes=input:8x3x320x320


# !!! process_engine.py contains workflow for build, export json, draw by trex tool.
# !!! use your path instead of above process_engine.py route
```