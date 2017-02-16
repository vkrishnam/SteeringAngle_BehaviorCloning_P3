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

additional = True #False

def generator(samples, folder_name, batch_size=32, corr=0.2):
    """
     Generator fucntion for the training pipeline
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
                name = folder_name+'/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # trim image to only see section with road
                # this would be done as part of the trimming layer in the model itself - as is needed while in driving (drive.py)

                #Also create the flipversion
                image_flipped = np.fliplr(center_image)
                angle_flipped = -1.0*center_angle

                images.append(image_flipped)
                angles.append(angle_flipped)

                #Also use the Left and Right Images
                correction = corr # this is a parameter to tune
                name = folder_name+'/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_angle = float(batch_sample[3]) + correction

                name = folder_name+'/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_angle = float(batch_sample[3]) - correction

                if additional is True:
                    images.append(left_image)
                    angles.append(left_angle)

                    images.append(right_image)
                    angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            if one_time == 0:
                #print("Input Image shape : ", center_image.shape)
                #print("X_train shape: ", X_train.shape)
                #print("y_train shape: ", y_train.shape)
                one_time = 1

            yield sklearn.utils.shuffle(X_train, y_train)


flags = tf.app.flags
FLAGS = flags.FLAGS


# command line flags
flags.DEFINE_string('data_folder',         '', "Folder where training data from the simulator is.")
flags.DEFINE_string('model_file',  'model.h5', "File name of the model generated or to be generated.")
flags.DEFINE_string('num_epoch',            5, "Number of epoch to train the model.")
flags.DEFINE_string('angle_correction',   0.1, "Parameter to tune the sieering looking from the Left/Right images.")

def steering_angle_prediction_model_description():
    """
    Model Description - Steering Angle prediction
    """
    # Model Description
    #  Motivated from NVIDIA DaveNet/BB8 Architecture (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
    model = Sequential()
    # Trim the image as part of Preprocess Stage
    model.add(Cropping2D(cropping=((50,10), (0,0)), input_shape=(160,320,3)))
    # Preprocess incoming data, normalize centered around zero with small standard deviation
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(100, 320, 3), output_shape=(100, 320, 3)))
    # Conv1
    model.add(Convolution2D(24, 5, 5, name="conv1"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    # Conv2
    model.add(Convolution2D(36, 5, 5, name="conv2"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    # Conv3
    model.add(Convolution2D(48, 5, 5, name="conv3"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    # Conv4
    model.add(Convolution2D(64, 3, 3, name="conv4"))
    model.add(Activation('relu'))
    # Conv5
    model.add(Convolution2D(64, 3, 3, name="conv5"))
    model.add(Activation('relu'))
    # Flatten
    model.add(Flatten())
    # FC1
    model.add(Dense(1024, name="fc1"))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.5))
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



def main(_):
    """
    Main function
    """
    #print("Data folder is: ",FLAGS.data_folder)
    samples = []
    num_epochs = int(FLAGS.num_epoch)

    # Open the csv file
    with open(FLAGS.data_folder+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers

        for line in reader:
            samples.append(line)
    #print(samples)
    from sklearn.utils import shuffle
    samples = shuffle(samples)

    # split the dataset into training and validation
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print("Num training samples: ", len(train_samples))
    print("Num validation samples: ", len(validation_samples))

    # compile and train the model using the generator function
    train_generator      = generator(train_samples,      batch_size=32, folder_name = FLAGS.data_folder)#, corr=float(FLAGS.angle_correction))
    validation_generator = generator(validation_samples, batch_size=32, folder_name = FLAGS.data_folder)#, corr=float(FLAGS.angle_correction))

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
    if additional is True:
        total_train_samples = len(train_samples)*4
        total_valid_samples = len(validation_samples)*4


    #  Train the model with the generators creating training and validation batches
    model.fit_generator(train_generator, samples_per_epoch=total_train_samples,\
            validation_data=validation_generator, nb_val_samples=total_valid_samples, \
            nb_epoch=num_epochs)
    # saving the model - arch, weights
    print("Saving the model file .....")
    model.save(FLAGS.model_file)  # creates a HDF5 file 'model.h5'

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




