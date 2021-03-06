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

[image1]: ./artifacts/steeringAngleHistogramInDataSet.png "steeringAngleHistogramInDataSet"
[image2]: ./artifacts/steeringAngleHistogramAfterDroppingZeroSteeringSamplesInDataSet.png  "steeringAngleHistogramAfterDroppingZeroSteeringSamplesInDataSet.png"
[image3]: ./artifacts/steeringAngleHistogramInTrainingSuite.png "steeringAngleHistogramInTrainingSuite"
[image4]: ./artifacts/thisIsWhatFirstConvLayerSees.png "conv1 perception"
[image5]: ./artifacts/thisIsWhatSecondConvLayerSees.png "conv2 perception"
[image6]: ./artifacts/thisIsWhatThirdConvLayerSees.png "conv3 perception"
[image7]: ./artifacts/thisIsWhatFourthConvLayerSees.png "conv4 perception"
[image11]: ./artifacts/thisIsWhatFifthConvLayerSees.png "conv5 perception"
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

######_Data Engineering_
The dataset is analyzed as a distribution of steering angle by ploting a histogram to show a highly skewed distribution with maximum samples of zero steering angle.
So it was conclusive enough that the earlier training trails were dominated by the zero steering angle data where by model was not learning except to predict zero steering angle.
Hence the data samples in the dataset corresponding to zero steering angle are seperated out where only a certain percentage of those samples randomly added to the training dataset. And the remaining samples were made part of the earlier split validation dataset.
Likewise the model is exposed to uniform samples of steering angles, and also this way the validation loss or accuracy shows more straight correspondence to simulator behavior in autonomous mode.
The Center, Left and Right images in the training samples are fully expolited so as the model to learn to track extremities. This has been acheived by the angle correction paramter applied to left and right images. This parameter is identified to 0.25 by iterative mechanism

Still on track#1, the simulator was failing to steer properly near the muddy patch. For this the pre-processing step of RGB to HSV conversion is added. This step is added as a pre-processing step even in the drive.py so that same pipeline is replicated for the simulator too.

**Muddy patch section with out demarkation in Track#1**
![alt text][image10]
![](./artifacts/muddyPatchSection.png =320x160)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

The same model is tried out on the track#2 , where it failed pathetically. So one lap of good driving data is recorded from track#2 for training.
At first model is afresh trained with consolidated data for 3 epoch with the same training strategy of retaining only 5% of zero steering angle data.
Then the mode is fine tuned with track#2 data alone for 3 more epochs with training strategy of retaining only 3% of zero steering angle data.

The refined model could make simulator run like a dream on both the tracks.

Lets have a look into what different conv layers are perceiving.

Conv1
![alt text][image4]


Conv2
![alt text][image5]


Conv3
![alt text][image6]


Conv4
![alt text][image7]


Conv5
![alt text][image11]

Overall it  looks the conv layers are arriving at the drivable area in the image.

####2. Final Model Architecture

Once the model was able to run successfully on both the tracks the network pruning activity is taken up becuase the model was heavy with around 300M parameters.
The numbers of paramters involved in the whole network is anlayzed by plotting the network using Keras utility functions. And finally the following model architecture is arrived with take around 145K parameters without affecting the accuracy or the simulator run.

Here is a details of the architecture

**_Model details:_**

![alt text][image8]

####3. Creation of the Training Set & Training Process

The dataset provided by the Udacity is already capturing good driving behavior, and hence no new data is captured/recorded for the track#1. The images of all three cameras (Left, Center and Right) are exploited for the corner cases and even the scenarios of the driving behavior on the the edges of the track.

#####_Training process_

The dataset is not blindly used for training, when tried the model is seen not to be learn whole lot needed for the successfully driving on the track. Instead the dataset is massaged where a certain percentage of samples having zero steering angle is dropped from the training suite. Just allowing 5% of the samples with zero steering angle is arrived at after few iterations. Idea behind this is borrowed from the reading the [blog] (https://mez.github.io/deep%20learning/2017/02/14/mainsqueeze-the-52-parameter-model-that-drives-in-the-udacity-simulator/).

_Histogram of steering angle in the samples in dataset_
![alt text][image1]


######_Data Augmentation_

-_Image Flip:_
The samples selected for training are also flipped using OpenCV (both Images and steering angles) and appended into the training suite.

-_Left and Right Camera Images:_
For the samples selected for the training, the left and right camera images are also considered to included in the training suite after their steering angle is corrected by a factor of +/-0.25.
Also for these left and right camera images, the flipped versions are also created and included in the training suite.

The Udacity provided dataset for track#1 has overall 8035 samples, out of which 4630 samples of zero steering angle and 3675 samples of non-zero steering angle.

| Total Samples in Dataset | Samples with zero steering angle | Samples with non-zero steering angle |
|-------------------------:|---------------------------------:|-------------------------------------:|
| 8035                     | 4630                             | 3675                                 |

Here is the histogram of steering angles in the dataset. We can notice how skewed is the distribution.
![alt text][image1]


After dropping the samples of zero steering angles by 95% randomly, we naturally end up with

| Total Samples in Dataset | Samples considered for training  | Samples considered for validation    |
|-------------------------:|---------------------------------:|-------------------------------------:|
| 8035                     | 3115                             | 4920                                 |


Here is the histogram of steering angles after dropping 95% of zero steering angle samples in the dataset. Still the distribution is not even.
![alt text][image2]

With the above sample division we end with following number of images for the training (center + 5 (center_flip,left, left_flip, right, right_flip) augmeted images) and validation suite:

| Total images in Validation | Total images in Training  |
|---------------------------:|--------------------------:|
| 4920                       | 18690                     |

Here is the histogram of steering angles in the training suite with augmentation. We can know how that the distribution is better even, but still lot of scope to cover the entire range.
![alt text][image3]

Training this dataset for 3 epochs ends in the following range of loss and accuracy.
```
18690/18690 [==============================] - 102s - loss: 0.0407 - acc: 0.0184 - val_loss: 0.0161 - val_acc: 0.8510
Epoch 2/3
18690/18690 [==============================] - 102s - loss: 0.0290 - acc: 0.0185 - val_loss: 0.0173 - val_acc: 0.8510
Epoch 3/3
18690/18690 [==============================] - 101s - loss: 0.0279 - acc: 0.0184 - val_loss: 0.0191 - val_acc: 0.8508
```
Point to note here is that, the validation suite, which has just the center images which would be only fed during the autonomous run, has a high accurary of ~85% which shows correspondence to the test driving success on the track.
**NOTE: Use of an adam optimizer makes the tuning of the learning rate unnecessary.**

Later for the track#2, one lap data of good driving behavior is recorded and merge with the original dataset from the Udacity.
Following the above described approach we end up in

| Total Samples in Dataset | Samples with zero steering angle | Samples with non-zero steering angle |
|-------------------------:|---------------------------------:|-------------------------------------:|
| 10310                    | 5720                             | 4590                                 |


After dropping the samples of zero steering angles by 95% randomly, we naturally end up with

| Total Samples in Dataset | Samples considered for training  | Samples considered for validation    |
|-------------------------:|---------------------------------:|-------------------------------------:|
| 10310                    | 3901                             | 6409                                 |

With the above sample division we end with following number of images for the training (center + 5 (center_flip,left, left_flip, right, right_flip) augmeted images) and validation suite:

| Total images in Validation | Total images in Training  |
|---------------------------:|--------------------------:|
| 6409                       | 23406                     |

Training this dataset for 3 epochs ends in the following range of loss and accuracy.
```
23406/23406 [==============================] - 129s - loss: 0.0826 - acc: 0.0294 - val_loss: 0.0544 - val_acc: 0.8084
Epoch 2/3
23406/23406 [==============================] - 128s - loss: 0.0650 - acc: 0.0328 - val_loss: 0.0524 - val_acc: 0.8064
Epoch 3/3
23406/23406 [==============================] - 129s - loss: 0.0637 - acc: 0.0329 - val_loss: 0.0492 - val_acc: 0.8070
```
Another 3 epochs of further fine tuning is done with the track#2 data only, which further reinforces the track#2 details into model and also to compensate for the lack of enough data samples as compared to the track#1 data.

The final model is able to successfully run on both tracks with the speed set to 9 mph.
Also to note, the simulator window dimensions had to kept at minimum for the lack of enough PC resources.

#####_Main Overall Insights gained from the project_
Dataset (both collection and engineering therafter) and training strategy is very key for the overall success of the project, than the model architecture and coding.

