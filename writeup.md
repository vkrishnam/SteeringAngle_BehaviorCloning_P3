#**Behavioral Cloning**

##Writeup

### Although the project submission mandates a writeup file, this file is used as journal and project notes to track/manage the tasks for successfull completion of the project.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* [X] Build, a convolution neural network in Keras that predicts steering angles from images
* [X] Train and validate the model with a data already shared
* [X] Sanity test of the model with Simulator
* [X] Use the simulator to collect data of good driving behavior
* [X] Test that the model successfully drives around track one without leaving the road
* [X] Focus on collecting more data with second track to make model generalize better
* [X] Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### I have given due consideration to the [rubric points](https://review.udacity.com/#!/rubrics/432/view) of the project individually and would be describing how those are addressed in the implementation.

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py - containing the script to create and train the model
* drive.py - for driving the car in autonomous mode
* model.h5 - containing a trained convolution neural network
* writeup.md - summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. Model archiitecture

Although initial Model experimented is highly inspired from NVIDIA end-to-end model architecture. Later it has been pruned to the following to reduce the complexity, size, paramters involved and training/inference times of the model.

For the model articulated in Keras, see the function
'''
steering_angle_prediction_model_description()
'''
The initial stages of cropping and normalization of pixels has been made as part of the model for two reasons, as those be needed even while model is deployed and other reason being as those operations being done by GPU during training and increasing the training time.

====================================================================================================
cropping2d_1 (Cropping2D)        (None, 60, 320, 3)    0           cropping2d_input_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 60, 320, 3)    0           cropping2d_1[0][0]
____________________________________________________________________________________________________
conv0 (Convolution2D)            (None, 60, 320, 8)    32          lambda_1[0][0]
____________________________________________________________________________________________________
conv1 (Convolution2D)            (None, 60, 320, 16)   1168        conv0[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 30, 160, 16)   0           conv1[0][0]
____________________________________________________________________________________________________
conv2 (Convolution2D)            (None, 30, 160, 8)    1160        maxpooling2d_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 15, 80, 8)     0           conv2[0][0]
____________________________________________________________________________________________________
conv3 (Convolution2D)            (None, 15, 80, 4)     292         maxpooling2d_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 7, 40, 4)      0           conv3[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1120)          0           maxpooling2d_3[0][0]
____________________________________________________________________________________________________
fc2 (Dense)                      (None, 128)           143488      flatten_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 128)           0           fc2[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 128)           0           activation_1[0][0]
____________________________________________________________________________________________________
fc3 (Dense)                      (None, 10)            1290        dropout_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 10)            0           fc3[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             11          activation_2[0][0]
====================================================================================================


####2. Attempts to reduce overfitting in the model

The model contains dropout layers inbetween FullyConnected layers and MaxPolling inbetween Conv layers in order to reduce overfitting.
The model was trained and validated on different data sets to ensure that the model was not overfitting.
See the function
'''
massageTheData(samples, percentage=0.9)
'''
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
Other parameters for the training of the model are
num_epoch to 3
angle correction used for left and right cameras is 0.25
percentage of data of zero steering angle to be dropped is 0.95

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
