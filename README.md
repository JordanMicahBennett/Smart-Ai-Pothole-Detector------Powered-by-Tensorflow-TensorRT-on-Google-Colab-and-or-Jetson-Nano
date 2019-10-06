
![Alt text](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/blob/master/data/display_2.png "default page")

# Smart (Ai) Pothole Detector (Powered by "Tensorflow/TensorRT" on "Google Colab" and or "Jetson Nano" via a Convolutional Artificial Neural Network)

![Alt text](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/blob/master/data/display.png "default page")

# Author

Jordan Bennett ([Website](folioverse.appspot.com)).

Thanks Google, TensorRt creators, thanks jhasuman, for his [desktop-version yolo-v2 based pothole detector](https://github.com/jhasuman/potholes-detection).

*   This project by Jordan essentially converts jhasuman's neural network based desktop pothole detector above (fp32 aka single precision floating point/32 bits), to jetson nano neural network based pothole detector (fp16 half precision floating point 16 bits). (**Purpose of which is to add the jetson nano with the trained half precision pothole detector to my car, and perhaps offer to others for sale?**)

*   In addition, the first 4 steps in the Alternative Instructions below were added by Jordan, and other steps modified to align with custom pothole model, to this [original blog/colab code](https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/).


# Success
There were lots of head scratching moments, but the tensorRT/jetson nano-mini computer version works fine, with seemingly similar accuracy to full Desktop version, as seen [in Part B/4 Prediction](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=ZrHyjN_Cvk4Z&line=14&uniqifier=1).


# Background 
Why do this? There is a lot to say about how damaging surprise potholes can be while driving, but instead I will leave this nice quick to the point summary here: "[Youtube/Hitting a pothole in a Tesla costs 2600 US dollars](https://www.youtube.com/watch?v=H6sPc9dFsGw)". This doesn't only happen to teslas either!


![Alt text](youtube_screenshot.jpg "default page")


That said, this Google Colab code is separate from the final product code I prepared for the jetson nano, although the nano code uses some of this colab code. 

The jetson nano is a portable device, and hence this may be attached to a vehicle to do pothole detection, based on [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network).

# I. Instructions to run on Jetson nano neural computer
1. Follow these instructions from [this Jetson Nano purchase and setup repository of mine](https://github.com/JordanMicahBennett/live_ai_object-detection-on-tiny-jetson-neural-nano-computer).

2. Download "[optimized trt_pothole_graph.pb graph](https://drive.google.com/file/d/1b9XgpXeWBay6GE2bnLSqlLSXDEFfUCZd/view?usp=sharing)" aka saved pothole detection neural network to somewhere on your jetson nano.

3. Download "[Ai Vehicle Pothole Detector (Powered by Jetson Nano Neural Computer)__________________.zip](https://drive.google.com/open?id=1wnO4IFE33CAppRkr0RI5TSgqU99J-wHO)" to somewhere on your jetson nano.

4. Copy .pb file from (2) to extracted directory of folder from (3) above.

5. Run "load_trt_graph.py" from (3).

6. Run jetson_nano_pothole_detector from (3), and see what jetson nano returns from the saved neural network.





# II. Alternatively, Instructions/steps to run on [Google Colab](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=Hwpja-Up3TU6&line=20&uniqifier=1) in your browser, if you don't have a jetson nano device.

The first 4 steps below were added by Jordan, and other steps added/modified to align with custom pothole model, based on this [original blog/colab code](https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/).

## Part A: Prequisites

0. [Connect to Jordan's Google drive to access saved neural nwtwork wrights etc](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=ma8JcJc9pzmH&line=11&uniqifier=1)

1. [Backend](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=qWBtb7Xin-zG&line=25&uniqifier=1)

2. [Utils](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=EGRugrAUnNrv&line=7&uniqifier=1)

3. [Frontend](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=lOgZDSUYnDQC&line=11&uniqifier=1)


## Part B: TensorRT Conversion & Usage steps

1. [Frozen graph creation](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=CgVxdMRCmFcn&line=10&uniqifier=1)

2. [TensorRT graph conversion of frozen graph](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=fWygvIyctpeI&line=10&uniqifier=1)

3. [Load tensor rt graph](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=L-Jx1Yq0uejv&line=4&uniqifier=1)

4. [Use loaded tensor rt graph to make predictions](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=ZrHyjN_Cvk4Z&line=14&uniqifier=1)


## Part C: Quick Test Order (I use the order below to run files in to run pothole prediction test, based on how I organized all files in this google colab project and on my google drive)

*   [Part A  (0)](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=ma8JcJc9pzmH&line=11&uniqifier=1) ----> [Part B (2b)](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=kGqN3UXquW-m) ----> [Part B (3a)](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=L-Jx1Yq0uejv&line=4&uniqifier=1) ----> [Part B (3b)](https://colab.research.google.com/drive/1kGV8DXJ7RwQtCDmd2QOc80Bll5n24Ftp#scrollTo=Azhh5OA2vI72&line=7&uniqifier=1) ----> [Part B (4)]()


# Performance comparison, between Desktop and TensorRT/Nano version:

1. See screenshot of fps count using TensorRT/Nano neural network pothole detector: https://drive.google.com/file/d/1LoWDsX75ehQ7HwcL1asz_EnRTZfHvrEs/view?usp=sharing

![Alt text](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/blob/master/data/JetsonNano_TensorRT%20Pothole%20Detector%20FPS%20Report%20(fp16_half_precision).jpg "default page")

2. See screenshot of fps count using Desktop neural network pothole detector: 
https://drive.google.com/file/d/1xnp304UfWpSWSLvqLvGDNGTHPFyBh9FN/view?usp=sharing 

![Alt text](https://github.com/JordanMicahBennett/Smart-Ai-Pothole-Detector------Powered-by-Tensorflow-TensorRT-on-Google-Colab-and-or-Jetson-Nano/blob/master/data/Desktop%20Pothole%20Detector%20FPS%20Report%20(fp32_single_floating_point).jpg "default page")





