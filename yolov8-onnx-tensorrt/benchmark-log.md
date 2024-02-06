### yolov8s.pt fp32 

<details close>
<summary>click to expand</summary>  

```bash
$ yolo detect val model=yolov8s.pt

WARNING ⚠️ 'data' is missing. Using default 'data=coco8.yaml'.
Ultralytics YOLOv8.1.9 🚀 Python-3.10.13 torch-1.13.1 CUDA:0 (NVIDIA GeForce RTX 3060, 12045MiB)
YOLOv8s summary (fused): 168 layers, 11156544 parameters, 0 gradients, 28.6 GFLOPs
val: Scanning /home/****/workspace/yolov8-tensorrt-comp/datasets/coco8/labels/val.cache... 4 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4/4 [00:00<?, ?it/s]
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  1.57it/s]
                  all          4         17      0.871       0.92      0.947      0.719
                person          4         10      0.854      0.587      0.707      0.379
                  dog          4          1      0.889          1      0.995      0.796
                horse          4          2      0.795          1      0.995        0.8
              elephant          4          2          1       0.93      0.995       0.55
              umbrella          4          1      0.686          1      0.995      0.895
          potted plant          4          1          1          1      0.995      0.895
Speed: 0.3ms preprocess, 6.6ms inference, 0.0ms loss, 1.4ms postprocess per image
Results saved to runs/detect/val4
💡 Learn more at https://docs.ultralytics.com/modes/val
```

</details>


### onnx fp32
<details close>
<summary>click to expand</summary>  

#### export
```bash
(yolov8-tensorrt) ****@MS7c90:~/workspace/yolov8-tensorrt-comp/ultralytics$ yolo export model=yolov8s.pt format=onnx
Ultralytics YOLOv8.1.9 🚀 Python-3.10.13 torch-1.13.1 CPU (AMD Ryzen 9 5950X 16-Core Processor)
YOLOv8s summary (fused): 168 layers, 11156544 parameters, 0 gradients, 28.6 GFLOPs

PyTorch: starting from 'yolov8s.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 84, 8400) (21.5 MB)

ONNX: starting export with onnx 1.12.0 opset 16...
ONNX: export success ✅ 0.6s, saved as 'yolov8s.onnx' (42.8 MB)

Export complete (2.1s)
Results saved to /home/****/workspace/yolov8-tensorrt-comp/ultralytics
Predict:         yolo predict task=detect model=yolov8s.onnx imgsz=640  
Validate:        yolo val task=detect model=yolov8s.onnx imgsz=640 data=coco.yaml  
Visualize:       https://netron.app
💡 Learn more at https://docs.ultralytics.com/modes/export
```
#### val
```bash
(yolov8-tensorrt) ****@MS7c90:~/workspace/yolov8-tensorrt-comp/ultralytics$ yolo detect val model=yolov8s.onnx
WARNING ⚠️ 'data' is missing. Using default 'data=coco8.yaml'.
Ultralytics YOLOv8.1.9 🚀 Python-3.10.13 torch-1.13.1 CUDA:0 (NVIDIA GeForce RTX 3060, 12045MiB)
Loading yolov8s.onnx for ONNX Runtime inference...
Forcing batch=1 square inference (1,3,640,640) for non-PyTorch models
val: Scanning /home/****/workspace/yolov8-tensorrt-comp/datasets/coco8/labels/val.cache... 4 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4/4 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00, 32.86it/s]
                   all          4         17      0.811      0.832      0.917      0.714
                person          4         10          1      0.494      0.693      0.377
                   dog          4          1      0.931          1      0.995      0.796
                 horse          4          2      0.844          1      0.995        0.8
              elephant          4          2      0.614        0.5      0.828      0.418
              umbrella          4          1       0.73          1      0.995      0.995
          potted plant          4          1      0.749          1      0.995      0.895
Speed: 4.7ms preprocess, 21.6ms inference, 0.0ms loss, 0.9ms postprocess per image
Results saved to runs/detect/val7
💡 Learn more at https://docs.ultralytics.com/modes/val

```

</details>

### onnx fp16(half)
<details close>
<summary>click to expand</summary>  

#### export
```bash
(yolov8-tensorrt) ****@MS7c90:~/workspace/yolov8-tensorrt-comp/ultralytics$ yolo export model=yolov8s.pt format=onnx device=0 half=true
Ultralytics YOLOv8.1.9 🚀 Python-3.10.13 torch-1.13.1 CUDA:0 (NVIDIA GeForce RTX 3060, 12045MiB)
YOLOv8s summary (fused): 168 layers, 11156544 parameters, 0 gradients, 28.6 GFLOPs

PyTorch: starting from 'yolov8s.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 84, 8400) (21.5 MB)

ONNX: starting export with onnx 1.12.0 opset 16...
ONNX: export success ✅ 0.4s, saved as 'yolov8s.onnx' (21.4 MB)

Export complete (3.7s)
Results saved to /home/****/workspace/yolov8-tensorrt-comp/ultralytics
Predict:         yolo predict task=detect model=yolov8s.onnx imgsz=640 half 
Validate:        yolo val task=detect model=yolov8s.onnx imgsz=640 data=coco.yaml half 
Visualize:       https://netron.app
💡 Learn more at https://docs.ultralytics.com/modes/export

```
#### val
```bash
(yolov8-tensorrt) ****@MS7c90:~/workspace/yolov8-tensorrt-comp/ultralytics$ yolo val task=detect model=yolov8s.onnx device=0 half=true
WARNING ⚠️ 'data' is missing. Using default 'data=coco8.yaml'.
Ultralytics YOLOv8.1.9 🚀 Python-3.10.13 torch-1.13.1 CUDA:0 (NVIDIA GeForce RTX 3060, 12045MiB)
Loading yolov8s.onnx for ONNX Runtime inference...
Forcing batch=1 square inference (1,3,640,640) for non-PyTorch models
val: Scanning /home/****/workspace/yolov8-tensorrt-comp/datasets/coco8/labels/val.cache... 4 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4/4 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00, 40.60it/s]
                   all          4         17      0.816      0.831      0.916      0.713
                person          4         10          1      0.485      0.689      0.376
                   dog          4          1      0.932          1      0.995      0.796
                 horse          4          2      0.846          1      0.995        0.8
              elephant          4          2       0.63        0.5      0.828      0.418
              umbrella          4          1      0.733          1      0.995      0.995
          potted plant          4          1      0.753          1      0.995      0.895
Speed: 3.4ms preprocess, 16.8ms inference, 0.0ms loss, 1.3ms postprocess per image
Results saved to runs/detect/val14
💡 Learn more at https://docs.ultralytics.com/modes/val

```

</details>

### trt fp32 engine
<details close>
<summary>click to expand</summary>

#### export
```bash
(yolov8-tensorrt) ****@MS7c90:~/workspace/yolov8-tensorrt-comp/ultralytics$ yolo export model=yolov8s.pt format=engine
WARNING ⚠️ TensorRT requires GPU export, automatically assigning device=0
Ultralytics YOLOv8.1.9 🚀 Python-3.10.13 torch-1.13.1 CUDA:0 (NVIDIA GeForce RTX 3060, 12045MiB)
YOLOv8s summary (fused): 168 layers, 11156544 parameters, 0 gradients, 28.6 GFLOPs

PyTorch: starting from 'yolov8s.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 84, 8400) (21.5 MB)

ONNX: starting export with onnx 1.12.0 opset 16...
ONNX: export success ✅ 0.5s, saved as 'yolov8s.onnx' (42.8 MB)

TensorRT: starting export with TensorRT 8.6.1...
[02/06/2024-13:53:47] [TRT] [I] [MemUsageChange] Init CUDA: CPU +202, GPU +0, now: CPU 3454, GPU 1793 (MiB)
[02/06/2024-13:53:51] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1220, GPU +266, now: CPU 4750, GPU 2059 (MiB)
[02/06/2024-13:53:51] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
[02/06/2024-13:53:51] [TRT] [I] ----------------------------------------------------------------
[02/06/2024-13:53:51] [TRT] [I] Input filename:   yolov8s.onnx
[02/06/2024-13:53:51] [TRT] [I] ONNX IR version:  0.0.8
[02/06/2024-13:53:51] [TRT] [I] Opset version:    16
[02/06/2024-13:53:51] [TRT] [I] Producer name:    pytorch
[02/06/2024-13:53:51] [TRT] [I] Producer version: 1.13.1
[02/06/2024-13:53:51] [TRT] [I] Domain:           
[02/06/2024-13:53:51] [TRT] [I] Model version:    0
[02/06/2024-13:53:51] [TRT] [I] Doc string:       
[02/06/2024-13:53:51] [TRT] [I] ----------------------------------------------------------------
[02/06/2024-13:53:51] [TRT] [W] onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 84, 8400) DataType.FLOAT
TensorRT: building FP32 engine as yolov8s.engine
[02/06/2024-13:53:51] [TRT] [I] Graph optimization time: 0.0138035 seconds.
[02/06/2024-13:53:51] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[02/06/2024-13:55:10] [TRT] [I] Detected 1 inputs and 3 output network tensors.
[02/06/2024-13:55:10] [TRT] [I] Total Host Persistent Memory: 320480
[02/06/2024-13:55:10] [TRT] [I] Total Device Persistent Memory: 38912
[02/06/2024-13:55:10] [TRT] [I] Total Scratch Memory: 4608
[02/06/2024-13:55:10] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 12 MiB, GPU 261 MiB
[02/06/2024-13:55:10] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 192 steps to complete.
[02/06/2024-13:55:10] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 7.01262ms to assign 8 blocks to 192 nodes requiring 35635712 bytes.
[02/06/2024-13:55:10] [TRT] [I] Total Activation Memory: 35635200
[02/06/2024-13:55:10] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +6, GPU +58, now: CPU 6, GPU 58 (MiB)
TensorRT: export success ✅ 83.7s, saved as 'yolov8s.engine' (61.6 MB)

Export complete (86.8s)
Results saved to /home/****/workspace/yolov8-tensorrt-comp/ultralytics
Predict:         yolo predict task=detect model=yolov8s.engine imgsz=640  
Validate:        yolo val task=detect model=yolov8s.engine imgsz=640 data=coco.yaml  
Visualize:       https://netron.app
💡 Learn more at https://docs.ultralytics.com/modes/export
```
#### val
```bash
(yolov8-tensorrt) ****@MS7c90:~/workspace/yolov8-tensorrt-comp/ultralytics$ yolo detect val model=yolov8s.engine
WARNING ⚠️ 'data' is missing. Using default 'data=coco8.yaml'.
Ultralytics YOLOv8.1.9 🚀 Python-3.10.13 torch-1.13.1 CUDA:0 (NVIDIA GeForce RTX 3060, 12045MiB)
Loading yolov8s.engine for TensorRT inference...
[02/06/2024-13:56:08] [TRT] [I] Loaded engine size: 61 MiB
[02/06/2024-13:56:08] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +57, now: CPU 0, GPU 57 (MiB)
[02/06/2024-13:56:08] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +34, now: CPU 0, GPU 91 (MiB)
[02/06/2024-13:56:08] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
val: Scanning /home/****/workspace/yolov8-tensorrt-comp/datasets/coco8/labels/val.cache... 4 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4/4 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.97it/s]
                   all          4         17      0.811      0.833      0.917      0.714
                person          4         10          1      0.496      0.693      0.377
                   dog          4          1      0.932          1      0.995      0.796
                 horse          4          2      0.843          1      0.995        0.8
              elephant          4          2      0.611        0.5      0.828      0.418
              umbrella          4          1      0.729          1      0.995      0.995
          potted plant          4          1      0.749          1      0.995      0.895
Speed: 4.5ms preprocess, 5.7ms inference, 0.0ms loss, 3.3ms postprocess per image
Results saved to runs/detect/val3
💡 Learn more at https://docs.ultralytics.com/modes/val

```

</details>


### trt fp16(half)  
<details close>
<summary>click to expand</summary>

#### export  
```bash
(yolov8-tensorrt) ****@MS7c90:~/workspace/yolov8-tensorrt-comp/ultralytics$ yolo export model=yolov8s.pt format=engine half
WARNING ⚠️ TensorRT requires GPU export, automatically assigning device=0
Ultralytics YOLOv8.1.9 🚀 Python-3.10.13 torch-1.13.1 CUDA:0 (NVIDIA GeForce RTX 3060, 12045MiB)
YOLOv8s summary (fused): 168 layers, 11156544 parameters, 0 gradients, 28.6 GFLOPs

PyTorch: starting from 'yolov8s.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 84, 8400) (21.5 MB)

ONNX: starting export with onnx 1.12.0 opset 16...
ONNX: export success ✅ 0.5s, saved as 'yolov8s.onnx' (42.8 MB)

TensorRT: starting export with TensorRT 8.6.1...
[02/06/2024-14:10:45] [TRT] [I] [MemUsageChange] Init CUDA: CPU +202, GPU +0, now: CPU 3454, GPU 1793 (MiB)
[02/06/2024-14:10:49] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1220, GPU +266, now: CPU 4750, GPU 2059 (MiB)
[02/06/2024-14:10:49] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
[02/06/2024-14:10:49] [TRT] [I] ----------------------------------------------------------------
[02/06/2024-14:10:49] [TRT] [I] Input filename:   yolov8s.onnx
[02/06/2024-14:10:49] [TRT] [I] ONNX IR version:  0.0.8
[02/06/2024-14:10:49] [TRT] [I] Opset version:    16
[02/06/2024-14:10:49] [TRT] [I] Producer name:    pytorch
[02/06/2024-14:10:49] [TRT] [I] Producer version: 1.13.1
[02/06/2024-14:10:49] [TRT] [I] Domain:           
[02/06/2024-14:10:49] [TRT] [I] Model version:    0
[02/06/2024-14:10:49] [TRT] [I] Doc string:       
[02/06/2024-14:10:49] [TRT] [I] ----------------------------------------------------------------
[02/06/2024-14:10:49] [TRT] [W] onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 84, 8400) DataType.FLOAT
TensorRT: building FP16 engine as yolov8s.engine
[02/06/2024-14:10:49] [TRT] [I] Graph optimization time: 0.0201429 seconds.
[02/06/2024-14:10:49] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[02/06/2024-14:14:00] [TRT] [I] Detected 1 inputs and 3 output network tensors.
[02/06/2024-14:14:00] [TRT] [I] Total Host Persistent Memory: 364672
[02/06/2024-14:14:00] [TRT] [I] Total Device Persistent Memory: 1024
[02/06/2024-14:14:00] [TRT] [I] Total Scratch Memory: 0
[02/06/2024-14:14:00] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 27 MiB, GPU 261 MiB
[02/06/2024-14:14:00] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 122 steps to complete.
[02/06/2024-14:14:00] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 3.37706ms to assign 8 blocks to 122 nodes requiring 17818624 bytes.
[02/06/2024-14:14:00] [TRT] [I] Total Activation Memory: 17817600
[02/06/2024-14:14:00] [TRT] [W] TensorRT encountered issues when converting weights between types and that could affect accuracy.
[02/06/2024-14:14:00] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to adjust the magnitude of the weights.
[02/06/2024-14:14:00] [TRT] [W] Check verbose logs for the list of affected weights.
[02/06/2024-14:14:00] [TRT] [W] - 59 weights are affected by this issue: Detected subnormal FP16 values.
[02/06/2024-14:14:00] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +21, GPU +22, now: CPU 21, GPU 22 (MiB)
TensorRT: export success ✅ 195.6s, saved as 'yolov8s.engine' (24.3 MB)

Export complete (198.7s)
Results saved to /home/****/workspace/yolov8-tensorrt-comp/ultralytics
Predict:         yolo predict task=detect model=yolov8s.engine imgsz=640 half 
Validate:        yolo val task=detect model=yolov8s.engine imgsz=640 data=coco.yaml half 
Visualize:       https://netron.app
💡 Learn more at https://docs.ultralytics.com/modes/export

```
#### val  
```bash
(yolov8-tensorrt) ****@MS7c90:~/workspace/yolov8-tensorrt-comp/ultralytics$ yolo detect val model=yolov8s.engine
WARNING ⚠️ 'data' is missing. Using default 'data=coco8.yaml'.
Ultralytics YOLOv8.1.9 🚀 Python-3.10.13 torch-1.13.1 CUDA:0 (NVIDIA GeForce RTX 3060, 12045MiB)
Loading yolov8s.engine for TensorRT inference...
[02/06/2024-14:14:37] [TRT] [I] Loaded engine size: 24 MiB
[02/06/2024-14:14:37] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +21, now: CPU 0, GPU 21 (MiB)
[02/06/2024-14:14:37] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +17, now: CPU 0, GPU 38 (MiB)
[02/06/2024-14:14:37] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
val: Scanning /home/****/workspace/yolov8-tensorrt-comp/datasets/coco8/labels/val.cache... 4 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4/4 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.38it/s]
                   all          4         17      0.813       0.83      0.917      0.722
                person          4         10          1      0.482      0.694      0.378
                   dog          4          1      0.931          1      0.995      0.796
                 horse          4          2      0.843          1      0.995        0.8
              elephant          4          2       0.62        0.5      0.828      0.469
              umbrella          4          1      0.731          1      0.995      0.995
          potted plant          4          1       0.75          1      0.995      0.895
Speed: 3.0ms preprocess, 2.1ms inference, 0.0ms loss, 3.4ms postprocess per image
Results saved to runs/detect/val6
💡 Learn more at https://docs.ultralytics.com/modes/val
```

</details>
