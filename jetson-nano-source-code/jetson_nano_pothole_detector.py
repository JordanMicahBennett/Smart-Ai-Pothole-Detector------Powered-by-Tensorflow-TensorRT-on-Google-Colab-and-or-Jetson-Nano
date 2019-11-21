###################
#Written by Jordan Bennett
#Part B/4. Make predictions based on FP16 graph. 
#This is not expected to work on windows.
#Expected to work on google colab or jetson nano like platform.
###################
from decode_hypothesis import decode_hypothesis
from load_trt_graph import image_size
from tensorflow.keras.preprocessing import image

def getPrediction (image_path__):
  img = image.load_img(image_path__, target_size=image_size[:2]) #where image_size[:2] = "[416,416,3]", which corresponds somewhat to config["input_size"] in config.json.
  
  x = image.img_to_array(img)/255.0 #CRUCIAL!!!-->jordan_normalize IMAGE_DATA=image.img_to_array(...) as seen in desktop version. Otherwise image data contains large integers, which is not expected by the trained pothole model which expects small normalized floating point values.
  x = np.expand_dims(x, axis=0)

  #x = preprocess_input(x) #irrelevant to hyppothesis accuracy
  
  feed_dict = {
      input_tensor_name: x
  }
  
  hypothesis = tf_sess.run(output_tensor, feed_dict) 

  hypothesis = hypothesis.reshape ( 13, 13, 5, 6 ) #jordan_addition: correct network output shape based on observation of desktop output analysis

  
  #jordan_note: The output of the neural network is a bunch of pixels, or bounding boxes. Cardinality of those boxes equals pothole cardinality.
  print('Caution!', len(decode_hypothesis(hypothesis)), 'pothole(s) are detected ahead from input image: ', image_path__ )


########################################################   
########################################################   
#####Test on image sample 0, with 8 potholes
getPrediction ('pothole_sample_0.jpg')
#####Test on image sample 1, with 8 potholes
getPrediction ('pothole_sample_1.jpg')
#####Test on image sample 2, with 8 potholes
getPrediction ('pothole_sample_2.jpg')
#####Test on image sample 3, with 0 potholes
getPrediction ('pothole_negative_sample.jpg')



###############################################
#Runtime cost test. Test speed of prediction on optimized tensor rt graph 
#this same code is ran in Desktop version, which yielded (except for getPrediction which is swapped with desktop equivalent)
print("\n\n########\nExecution runtime cost test")
import time
times = []
for i in range(20):
    start_time = time.time()
    getPrediction ('pothole_sample_2.jpg')
    delta = (time.time() - start_time)
    times.append(delta)
mean_delta = np.array(times).mean()
fps = 1 / mean_delta
print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))
