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
[image8]: ./artifacts/model.png "Model Architecture"
[image9]: ./artifacts/modelIlustration.png "Model Architecture Illustrated"
[image10]: ./artifacts/muddyPatchSection.png "Muddy Patch section"

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

####1. Model Architecture

Although initial Model experimented is highly inspired from NVIDIA end-to-end model architecture. Later it has been pruned to the following to reduce the complexity, size, paramters involved and training/inference times of the model.

For the model articulated in Keras, see the function
```python
steering_angle_prediction_model_description()
```
The initial stages of cropping and normalization of pixels has been made as part of the model for two reasons, as those be needed even while model is deployed and other reason being as those operations being done by GPU during training and increasing the training time.

**_Model Illustration_**
![alt text][image9]

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in between FullyConnected layers and MaxPolling in between Conv layers in order to reduce overfitting.
The model was trained and validated on different data sets to ensure that the model was not overfitting.
See the function
```python
massageTheData(samples, percentage=0.9)
```
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
Other parameters for the training of the model are
- num_epoch to 3
- angle correction used for left and right cameras is 0.25
- percentage of data of zero steering angle to be dropped is 0.95

####4. Appropriate training data

To start with the data provided by Udacity is taken and trained the model. This dataset captures good driving behavior and also helpful in gaining initial insight into the kind of Center, Left and Right images, how much of correction is needed and also to analyze the distribution of steering angles.

Actually for track#1, no new  training data is used, although new training dataset was recorded, by following a training and validation strategy which is described next.

Training data fully exploited the  combination of center camera, left and right camera data to recover from the left and right sides of the road.

For details about how the training data created, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The initial implementation of the model was almost same one as NVIDIA end-to-end learning architecture, with additional MaxPooling and Dropout layers just to avoid overfitting in case, except for the initial layers of cropping and pixel normalization and the changed sizes there of.

This model might be appropriate because as the CNN digests the scene interms of convolution layers and regressing (done by fc layers) a floating point number which is the steering angle in our case.

When trained on the data set with a blind train/validation split of 80:20, the both loss and accuracy were good both on training and validation set. But the simulator was not able to run in autonomous mode, but more it was going in a straight line becuase of steering anlge prediction hovering around zero value. Then more trails were done with the model to find that this behavior has nothing to do with model but more to do with the dataset and the pre-processing/data engineering of it.

######Data Engineering
The dataset is analyzed as a distribution of steering angle by ploting a histogram to show a highly skewed distribution with maximum samples of zero steering angle.
So it was conclusive enough that the earlier training trails were dominated by the zero steering angle data where by model was not learning except to predict zero steering angle.
Hence the data samples in the dataset corresponding to zero steering angle are seperated out where only a certain percentage of those samples randomly added to the training dataset. And the remaining samples were made part of the earlier split validation dataset.
Likewise the model is exposed to uniform samples of steering angles, and also this way the validation loss or accuracy shows more straight correspondence to simulator behavior in autonomous mode.
The Center, Left and Right images in the training samples are fully expolited so as the model to learn to track extremities. This has been acheived by the angle correction paramter applied to left and right images. This parameter is identified to 0.25 by iterative mechanism

Still on track#1, the simulator was failing to steer properly near the muddy patch. For this the pre-processing step of RGB to HSV conversion is added. This step is added as a pre-processing step even in the drive.py so that same pipeline is replicated for the simulator too.
**Muddy patch section with out demarkation in Track#1**
![alt text][image10]


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

The same model is tried out on the track#2 , where it failed pathetically. So one lap of good driving data is recorded from track#2 for training.
At first model is afresh trained with consolidated data for 3 epoch with the same training strategy of retaining only 5% of zero steering angle data.
Then the mode is fine tuned with track#2 data alone for 3 more epochs with training strategy of retaining only 3% of zero steering angle data.

The refined model could make simulator run like a dream on both the tracks.

####2. Final Model Architecture

Once the model was able to run successfully on both the tracks the network pruning activity is taken up becuase the model was heavy with around 300M parameters.
The numbers of paramters involved in the whole network is anlayzed by plotting the network using Keras utility functions. And finally the following model architecture is arrived with take around 145K parameters without affecting the accuracy or the simulator run.

Here is a details of the architecture

**_Model details:_**
![alt text][image8]

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
