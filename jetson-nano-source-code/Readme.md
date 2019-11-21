
# Instructions to run "jetson-nano-source-code" on Jetson nano neural computer

1. Follow these instructions from [this Jetson Nano purchase and setup repository of mine](https://github.com/JordanMicahBennett/live_ai_object-detection-on-tiny-jetson-neural-nano-computer).

2. Download [this repository](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/), and open the "[jetson-nano-source-code](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/tree/master/jetson-nano-source-code)" folder.

3. Download "[optimized trt_pothole_graph.pb graph](https://drive.google.com/file/d/1b9XgpXeWBay6GE2bnLSqlLSXDEFfUCZd/view?usp=sharing)" aka saved pothole detection neural network to somewhere on your jetson nano.

4. Copy .pb file from (3) to extracted directory of folder from (2) above.

5. Run "[load_trt_graph.py](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/blob/master/jetson-nano-source-code/load_trt_graph.py)" from "[jetson-nano-source-code](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/tree/master/jetson-nano-source-code)" folder from item (2).

6. Run "[jetson_nano_pothole_detector.py](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/blob/master/jetson-nano-source-code/jetson_nano_pothole_detector.py)" from (5), and see what jetson nano returns from the saved neural network.

7. Enjoy!
