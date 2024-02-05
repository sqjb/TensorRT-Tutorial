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
```

### convert to TRT engine
```bash
$ trtexec --explicitBatch --onnx='./yolov8s-dy-input.onnx' --saveEngine=test.engine
```

### inference test with TRT engine
```bash
$ trtexec --loadEngine=test.engine --batch=1
```
