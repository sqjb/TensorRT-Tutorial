## yolov8->onnx->tensorrt engine test 
### installation
1. step-1  make sure tensorrt has been installed on your host.
2. step-2 :install yolov8  
  2-1. install pytorch via conda  
  2-2. install yolov8 by cmd ``$ pip install ultralytics``  
  2-3. varify by  ``$ yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'``


### convert to ONNX 
```bash
$ yolo export model=yolov8s.pt format=onnx dynamic=true
# change input tensor name from 'images' -> 'input'
$ python chg-input-name.py 
```

### convert to TRT engine
```bash
$ trtexec --explicitBatch --onnx='./yolov8s-dy-input.onnx' --saveEngine=test.engine
```

### inference test with TRT engine
```bash
$ trtexec --loadEngine=test.engine --batch=1
```

### benchmarks
> rtx 3060, ubuntu 20.04, cuda toolkit 11.6.2, tensorrt 8.6, python 3.10  

| cmd  | inference time  |
|---|---|
| yolo predict model=yolov8s.pt source=images  | 6.2ms  |
| yolo predict model=yolov8s.pt source=images half=true | 15.1 ms |
| yolo predict model=yolov8s.pt source=images int8=true | 6.2 ms  |
| trtexec --explicitBatch --onnx=./yolov8s-dy-input.onnx --saveEngine=test.engine --shapes=input:8x3x320x320 | 9.3 ms  |
| trtexec --explicitBatch --onnx=./yolov8s-dy-input.onnx --saveEngine=test.engine --shapes=input:8x3x320x320 –fp16 | 3.9 ms  |
| trtexec --explicitBatch --onnx=./yolov8s-dy-input.onnx --saveEngine=test.engine --shapes=input:8x3x320x320 --int8 | 2.8 ms  |

### issue
https://github.com/ultralytics/ultralytics/issues/1719
