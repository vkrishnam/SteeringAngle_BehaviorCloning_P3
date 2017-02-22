import numpy as np
import os
import csv
import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model



import cv2
import numpy as np
import sklearn

from keras.models import Sequential, Model
from keras.layers import Cropping2D

def dummyGenerator(samples, aug=False):
    """
     Dummy Generator function for analyzing the training data
    """
    overall = []
    num_samples = len(samples)
    one_time = 0
    batch_size = 32
    corr = 0.25
    batch_size = batch_size//2
    angles = []
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            for batch_sample in batch_samples:

                #Center and Center Flip
                angle = float(batch_sample[3])
                angle_flipped = -1.0*angle
                angles.append(angle)
                if aug is True:
                    angles.append(angle_flipped)

                #Left and Left Flip
                correction = corr # this is a parameter to tune
                angle = float(batch_sample[3]) + correction
                angle_flipped = -1.0*angle
                if aug is True:
                    angles.append(angle)
                    angles.append(angle_flipped)

                #Right and Right Flip
                correction = corr # this is a parameter to tune
                angle = float(batch_sample[3]) - correction
                angle_flipped = -1.0*angle
                if aug is True:
                    angles.append(angle)
                    angles.append(angle_flipped)


        #Analyze the steering angles
        #overall = np.array(angles)
        overall_steering_angles = np.array(angles)
        steering_hist, edges = np.histogram(overall_steering_angles, bins=50)
        if aug is True:
            print("Hist of training samples with Augmentation")
        else:
            print("Hist of training samples without Augmentation")

        print(steering_hist)
        break
    return 0



def generator(samples, folder_name, batch_size=32, corr=0.2):
    """
     Generator function for the training pipeline
    """
    num_samples = len(samples)
    one_time = 0
    batch_size = batch_size//2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = folder_name+'/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    #1
                    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

                    # trim image to only see section with road
                    # this would be done as part of the trimming layer in the model itself - as is needed while in driving (drive.py)

                    #Also create the flipversion
                    image_flipped = np.fliplr(image)

                    #2
                    images.append(cv2.cvtColor(image_flipped, cv2.COLOR_BGR2HSV))
                #Center and Center Flip
                angle = float(batch_sample[3])
                angle_flipped = -1.0*angle
                angles.append(angle)
                angles.append(angle_flipped)

                #Left and Left Flip
                correction = corr # this is a parameter to tune
                angle = float(batch_sample[3]) + correction
                angle_flipped = -1.0*angle
                angles.append(angle)
                angles.append(angle_flipped)

                #Right and Right Flip
                correction = corr # this is a parameter to tune
                angle = float(batch_sample[3]) - correction
                angle_flipped = -1.0*angle
                angles.append(angle)
                angles.append(angle_flipped)


            X_train = np.array(images)
            y_train = np.array(angles)

            if one_time == 0:
                #print("Input Image shape : ", center_image.shape)
                #print("X_train shape: ", X_train.shape)
                #print("y_train shape: ", y_train.shape)
                one_time = 1

            yield sklearn.utils.shuffle(X_train, y_train)


def generator2(samples, folder_name, batch_size=32, corr=0.2):
    """
     Generator function for the validation pipeline
    """
    num_samples = len(samples)
    one_time = 0
    batch_size = batch_size//2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(1):
                    name = folder_name+'/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    #1
                    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

                #Center and Center Flip
                angle = float(batch_sample[3])
                angles.append(angle)

            X_valid = np.array(images)
            y_valid = np.array(angles)

            if one_time == 0:
                #print("Input Image shape : ", center_image.shape)
                #print("X_train shape: ", X_train.shape)
                #print("y_train shape: ", y_train.shape)
                one_time = 1

            yield sklearn.utils.shuffle(X_valid, y_valid)


flags = tf.app.flags
FLAGS = flags.FLAGS


# command line flags
flags.DEFINE_string('data_folder',         '', "Folder where training data from the simulator is.")
flags.DEFINE_string('model_file',  'model.h5', "File name of the model generated or to be generated.")
flags.DEFINE_string('num_epoch',            5, "Number of epoch to train the model.")
flags.DEFINE_string('angle_correction',   0.2, "Parameter to tune the sieering looking from the Left/Right images.")
flags.DEFINE_string('percentage_zero',   0.9, "Parameter to choose the percentage of zero steering samples to drop.")

def steering_angle_prediction_model_description():
    """
    Model Description - Steering Angle prediction
    """
    # Model Description
    #  Motivated from NVIDIA DaveNet/BB8 Architecture (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
    model = Sequential()
    # Trim the image as part of Preprocess Stage
    model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(160,320,3)))
    # Preprocess incoming data, normalize centered around zero with small standard deviation
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(60, 320, 3), output_shape=(60, 320, 3)))
    # Conv0
    model.add(Convolution2D( 8, 1, 1, activation='relu', border_mode='same', name="conv0"))
    # Conv1
    model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', name="conv1"))
    model.add(MaxPooling2D((2, 2)))
    # Conv2
    model.add(Convolution2D( 8, 3, 3, activation='relu', border_mode='same', name="conv2"))
    model.add(MaxPooling2D((2, 2)))
    # Conv3
    model.add(Convolution2D( 4, 3, 3, activation='relu', border_mode='same', name="conv3"))
    model.add(MaxPooling2D((2, 2)))
    # Conv4
    #model.add(Convolution2D(64, 3, 3, activation='relu', name="conv4"))
    # Conv5
    #model.add(Convolution2D(64, 3, 3, activation='relu', name="conv5"))
    # Flatten
    model.add(Flatten())
    # FC1
    #model.add(Dense(1024, name="fc1"))
    #model.add(Activation('relu'))
    # Dropout
    #model.add(Dropout(0.5))
    # FC2
    model.add(Dense(128, name="fc2"))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.5))
    # FC3
    model.add(Dense(10, name="fc3"))
    model.add(Activation('relu'))
    #Final
    model.add(Dense(1))
    return model

def plotHistogram1(samples):
    steering_measurement = []
    for sample in samples:
        steering_measurement.append(float(sample[3]))

    steering_measurement = np.array(steering_measurement)
    steer_hist, binEdges= np.histogram(steering_measurement, bins=50)
    print(steer_hist)
    print(binEdges)
    print("BinEdge : Population")
    for edge, bi in zip(binEdges, steer_hist):
        print("{} : {}".format(str(edge), str(bi)))

    return 0


def massageTheData(samples, percentage=0.9):
    """
    Rationale of this function is to selectively drop the zero steering angle data as talked about in the below blog.
    https://mez.github.io/deep%20learning/2017/02/14/mainsqueeze-the-52-parameter-model-that-drives-in-the-udacity-simulator/
    """
    from sklearn.utils import shuffle
    remin = 1 - percentage
    indexWhereSteeringIsZero = shuffle([i for i in range(len(samples)) if float(samples[i][3]) == 0.0 ])
    indexWhereSteeringIsNonZero = shuffle([i for i in range(len(samples)) if float(samples[i][3]) != 0.0 ])
    print("Number of samples with ZERO steering angle : ", len(indexWhereSteeringIsZero))
    print("Number of samples with Non-ZERO steering angle : ", len(indexWhereSteeringIsNonZero))
    print("Total : ", len(indexWhereSteeringIsNonZero)+len(indexWhereSteeringIsZero))
    newSamples = [samples[i] for i in indexWhereSteeringIsNonZero]
    vestige = []
    for i in range(len(indexWhereSteeringIsZero)):
        if i < (remin*len(indexWhereSteeringIsZero)):
            newSamples.append(samples[indexWhereSteeringIsZero[i]])
        else:
            vestige.append(samples[indexWhereSteeringIsZero[i]])

    return newSamples, vestige




def main(_):
    """
    Main function

    Typical usage:
    python model.py --data_folder hillData/ --num_epoch 3 --angle_correction 0.25 --percentage_zero 0.97
    """
    #print("Data folder is: ",FLAGS.data_folder)
    samples = []
    num_epochs = int(FLAGS.num_epoch)

    per = float(FLAGS.percentage_zero)

    # Open the csv file
    with open(FLAGS.data_folder+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers

        for line in reader:
            samples.append(line)

    #Plot the histogram of data of steering angles
    plotHistogram1(samples)
    # Massage the data such that the districutions of samples is more even
    # Drop the incidence of ZERO steering angle by 90%
    samples, vestige_samples = massageTheData(samples, per)

    #print(samples)
    from sklearn.utils import shuffle
    samples = shuffle(samples)

    # split the dataset into training and validation
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    for i in range(len(vestige_samples)):
        validation_samples.append(vestige_samples[i])

    print("Num training samples: ", len(train_samples))
    print("Num validation samples: ", len(validation_samples))
    print("Total samples: ", len(train_samples)+len(validation_samples))

    # Generate Distributions of the steering angle
    #dummyGenerator(train_samples, aug=False)
    #dummyGenerator(train_samples, aug=True)

    # compile and train the model using the generator function
    train_generator      = generator(train_samples,      batch_size=32, folder_name = FLAGS.data_folder, corr=float(FLAGS.angle_correction))
    validation_generator = generator2(validation_samples, batch_size=32, folder_name = FLAGS.data_folder, corr=float(FLAGS.angle_correction))

    # Try to load the model
    #   if the model file does not exist then go and define the model
    #   else pick the trained model so far for further training
    try:
        model = load_model(FLAGS.model_file)
    except:
        print("model not found...hence defining the model afresh!!!!")
        # Model Description
        model = steering_angle_prediction_model_description()
        #  Compile the model and show the summary
        model.compile('adam', 'mse', ['accuracy'])
        model.summary()
        #from keras.utils.visualize_util import plot
        #plot(model, to_file='model.png')


    total_train_samples = len(train_samples)*2
    total_valid_samples = len(validation_samples)*2
    total_train_samples = len(train_samples)*6
    total_valid_samples = len(validation_samples)*6


    #  Train the model with the generators creating training and validation batches
    model.fit_generator(train_generator, samples_per_epoch=total_train_samples,\
            validation_data=validation_generator, nb_val_samples=total_valid_samples, \
            nb_epoch=num_epochs)
    # saving the model - arch, weights
    print("Saving the model file .....")
    model.save(FLAGS.model_file)  # creates a HDF5 file 'model.h5'

    # print the illustration of the model
    print("Printing the model...")
    from keras.utils.visualize_util import plot
    plot(model, "model.png", show_shapes=True)
    # lets find how good is the model on the validation and training data
    print("::::Validation data accuracy::::")
    metrics = model.evaluate_generator(validation_generator, total_valid_samples)
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))
    print("::::Train data accuracy::::")
    metrics = model.evaluate_generator(train_generator, total_train_samples)
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))

    del model


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()




