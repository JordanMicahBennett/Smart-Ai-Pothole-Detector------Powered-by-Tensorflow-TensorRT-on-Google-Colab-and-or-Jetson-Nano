
# Instructions to run "jetson-nano-source-code" on Jetson nano neural computer

1. Follow these instructions from [this Jetson Nano purchase and setup repository of mine](https://github.com/JordanMicahBennett/live_ai_object-detection-on-tiny-jetson-neural-nano-computer).

2. Download [this repository](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/), and open the "[jetson-nano-source-code](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/tree/master/jetson-nano-source-code)" folder, to your jetson nano device.

3. Download "[optimized trt_pothole_graph.pb graph](https://drive.google.com/file/d/1b9XgpXeWBay6GE2bnLSqlLSXDEFfUCZd/view?usp=sharing)" aka saved pothole detection neural network to somewhere on your jetson nano. [I had converted from a 32 bit single precision floating point pothole detector, to this 16 bit half precision .pb version using Tensor RT](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano#author).

4. Copy .pb file from (3) to extracted directory of folder from (2) above.

5. Install essential compile/lib files, (...otherwise face lots of scipy etc build errors at steps (6) and (7) below)

`sudo apt-get install -y build-essential libatlas-base-dev gfortran`

6. Install jetson compatible tensorflow 2.2.0 (...otherwise face a ValueError/ missing node problem, caused by mismatched tensorflow graph freezing/graph loading attempt)

`sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow`

7. Install jetson compatible  tensorflow gpu-2.0.0 (...otherwise face libcudart.so.10.0 and segmentation errors)

`sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow-gpu`

8. Install jupyter lab:

`sudo pip3 install jupyterlab`

9. Go to directory of pothole "[jetson-nano-source-code](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/blob/master/jetson-nano-source-code/)", and run jupyter lab:

`sudo jupyter lab --allow-root`

10. Finally, to perform prediction on the sample images, in jupyter lab tab, select "[jetson_nano_pothole_detector](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/blob/master/jetson-nano-source-code/jetson_nano_pothole_detector.ipynb)" notebook file, and hit "Run All" from run menu at top, and wait for pothole predictions of sample pothole image samples on jetson nano!!! 

11. Enjoy



