# Human-Activity-Recognition
Human Activity Recognition using OpenPose

Human Activity Recognition Using Deep Learning and Human Pose Estimation

Abstract

Human activity recognition is a challenging classification task which involves predicting the movement of a person based on some sensory data. Recently, deep learning methods such as convolutional neural networks and recurrent neural networks have shown capable and even achieve state-of-the-art results by automatically learning features from the raw sensor data [1]. This paper however, takes a more rudimentary approach to human activity recognition by using human pose estimation as an intermediary. Human pose estimation is a way of identifying and classifying the joints in the human body. Essentially it is a way to capture a set of coordinates for each joint (arm, head, torso, etc.,) which is known as a key point that can describe a pose of a person. The connection between these points is known as a pair. There are three types of approaches to model the human body: the skeleton-based model (the one used in this paper), the contour-based model, and the volume-based model [2]. We have used the prebuilt OpenPose demo pretrained model for human pose estimation, which is able to detect 25 key points [3], and overlaid it with a deep neural network to create the classifier for human activity recognition.
 
Methodology

The input data for our model is videos that consist of a minimum of 30 frames containing one individual doing a task of interest (such as running). These videos are run through the OpenPose model, which outputs JSON files containing the coordinates of each of the key points of the person along with their confidence levels, a measure of the model’s confidence in the key points it predicts [4]. Every JSON file corresponds to each frame, so a video containing at least 30 frames would output at least 30 JSON files. Since each JSON file contains 25 key points, each of which has an x-coordinate, a y-coordinate, and a confidence level, each JSON files contains 75 important pieces of data. Iterating over the 30 frames and appending each of their 75 pieces of data into a one-dimensional array of size 75*30 = 2250 gives us the input of one training example. The model output is a number which corresponds to an activity of interest, such as 0 being running, 1 being cycling, etc. The combination of a training input and output gives a full training example and accumulating multiple training examples from multiple videos gives the full training data set needed to train the neural network. The network itself is a quite standard multi-layered neural network which used mini-batch stochastic gradient descent with the sigmoid function as the activation function. The input layer has 2250 neurons and the output layer has as many neurons as there are activities to classify. 

Bibliography

[1] https://machinelearningmastery.com/deep-learning-models-for-human-activity-recognition/

[2] https://www.v7labs.com/blog/human-pose-estimation-guide#:~:text=Human%20Pose%20Estimation%20(HPE)%20is,is%20known%20as%20a%20pair.

[3] https://github.com/CMU-Perceptual-Computing-Lab/openpose

[4] https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md

# Training the Model and Getting it to run:

Configuring OpenPose:

Download either the CPU version or GPU version of the OpenPose Demo from the following link:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases

Follow the `Instructions.txt` file inside the downloaded zip file to download the models required by OpenPose (about 500 Mb).
Then, you can run OpenPose from the PowerShell command-line or CMD by following https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md.


Use the flag `--write_json folder_name\` with the video's to obtain the JSON files

use flag `--keypoint_scale 3` to scale the coordinates to the range [0,1]

```
Windows Example:

bin\OpenPoseDemo.exe --video examples/media/video.avi --write_json output_jsons/ --keypoint_scale 3
```

```
Ubuntu and Mac Example

./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_json output_jsons/ --keypoint_scale 3
```

This will give the desired JSON files in the desired folder. Iterating through every 30 JSON files and appending the 75 pieces of data into one 1D array will constitute one one training example.

# Collecting Training and Testing Data for the Model

The Output label of each training example will be a number that corresponds to the activity that's being detected. To change the amount of activities being detected, change the definition of `def vectorized_result(j)` to `e=np.zeros((number_of_activities,1))` in line 150 of [ActivityDetection.py](ActivityDetection.py)

Implement `def load_data()` in line 157 of [ActivityDetection.py](ActivityDetection.py) based on how the training and test data has been collected. The output of `load_data()` should be `(x_train, y_train), (x_test, y_test)`. x_train should be of dimension (number of training examples, 2250) where 2250 is because each frame has 25 keypoints which translates to 75 pieces of information, coupled with 30 frames. It's assumed the coordinates of x_train are normalized between [0,1]. y_train is expected to have dimension (number_of_training_examples, 1) where the y value of each training example is the number that corresponds to the activity. x_test and y_test are defined in the same way. 

Once the training and test data set is well defined and imported correctly, simply run the program to both train and test the accuracy of the neural network.

# Additional Information:
If the network does not train well under the training dataset, consider modifying the network parameters such as increasing the neurons in the hidden layers, increasing hidden layers, changing the learning rate, changing the mini-batch size, or changing the number of epochs to train for.

To change the number of neurons in a hidden layer or increase the number of hidden layers, change line 176 of [ActivityDetection.py](ActivityDetection.py). For more detail, check the definition of the `Network` class initializer starting on line 13 of [ActivityDetection.py](ActivityDetection.py).

To change the learning rate, mini-batch size, or number of epochs, change line 179 of [ActivityDetection.py](ActivityDetection.py). For more detail, check the definition of the `def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None)` on line 37 of [ActivityDetection.py](ActivityDetection.py).
