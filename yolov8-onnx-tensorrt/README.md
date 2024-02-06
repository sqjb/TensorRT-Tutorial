## yolov8->onnx->tensorrt engine test 
plz refer to my test env.  
> hardware: rtx-3060  
> os: utunbu 20.04  
> cuda-toolkit: 11.6.2  
> tensrrt: 8.6.1  
> python: 3.10
> yolov8: 8.1

### Installation
> Is't better to install yolv8 from source code since the **Compatibility Issues** in tensorrt and pytorch 
<details close>
<summary>click to expand</summary>

#### 1. install tensorrt  
refer to [tensorrt-helloworld](../tensorrt-helloworld/README.md)
#### 2. install tensorrt python
```bash
# create a new env for the example.
$ conda create -n yolov8-onnx-tensorrt python=3.10
$ conda activate yolov8-onnx-tensorrt
# install tensorr for yolov8-onnx-tensorrt conda env.
$ cd /usr/local/TensorRT-8.6.1.6/python/ 
$ python3 -m pip install tensorrt-8.6.1-cp310-none-linux_x86_64.whl
```
#### 3. install yolv8
##### 3-1. download repo
```bash
$ git clone https://github.com/ultralytics/ultralytics.git
```
##### 3-2. change ``nvidia-tensorrt`` requirement to ``tensorrt``
*  ``ultralytics/nn/autobackend.py:204``
```python
199             LOGGER.info(f"Loading {w} for TensorRT inference...")
200             try:
201                 import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
202             except ImportError:
203                 if LINUX:
204                     #check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
205                     check_requirements("tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
206                 import tensorrt as trt  # noqa
207             check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
208             if device.type == "cpu":
209                 device = torch.device("cuda:0")

```
* ``ultralytics/engine/exporter.py:635``
```python
 630 
 631         try:
 632             import tensorrt as trt  # noqa
 633         except ImportError:
 634             if LINUX:
 635                 #check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
 636                 check_requirements("tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
 637             import tensorrt as trt  # noqa
 638         
 639         check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
 640         
 641         self.args.simplify = True

```
<!-- ##### 3-2. comment ``meta`` part for build tenerrt engine
``ultralytics/engine/exporter.py:[689-691]``
```python
 687         with builder.build_engine(network, config) as engine, open(f, "wb") as t:
 688             # Metadata
 689             #meta = json.dumps(self.metadata)
 690             #t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
 691             #t.write(meta.encode())
 692             # Model
 693             t.write(engine.serialize())
 694 
 695         return f, None

``` -->
#### 4. install
```bash
$ cd ultralytics
$ pip install -e .
``` 
#### 5. verify
```bash
# predict
$ yolo predict model=yolov8s.pt source='https://ultralytics.com/images/bus.jpg'

# export trt
$ yolo export model=yolov8s.pt format=engine

# val
$ yolo detect val model=yolov8s.engine
```

</details>  

### benchmarks (COCO)
> rtx 3060, ubuntu 20.04, cuda toolkit 11.6.2, tensorrt 8.6, python 3.10 
> * export onnx and trt by [yolo export command](https://docs.ultralytics.com/modes/export/)

| model | command  | dtype  | inference-speed (ms) | mAP50 |  
|---|---|---|---|---|
| yolov8s.pt | yolo detect val model=yolov8s.pt | FP32 | 6.6 | 0.947 |
| yolov8s.onnx | yolo detect val model=yolov8s.onnx | FP32 | 22.7 | 0.917 |
| yolov8s-fp16.onnx | yolo detect val model=yolov8s.onnx devoce=0 half=true | FP16 | 16.8 | 0.916 |
| yolov8s-fp32.engine | yolo detect val model=yolov8s-fp32.onnx | FP32 | 5.7 | 0.917 |
| yolov8s-fp16.engine | yolo detect val model=yolov8s-fp16.onnx | FP16 | 2.1 | 0.917 |
check details >> [benchmark-log](./benchmark-log.md)

### issue
https://github.com/ultralytics/ultralytics/issues/1719
