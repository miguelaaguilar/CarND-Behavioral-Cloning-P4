import os
import csv
import numpy as np
import cv2
import sklearn
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D, BatchNormalization


data_dir = '/home/workspace/data/'
data_file1 = data_dir + 'driving_log1.csv' # Track 1
data_file2 = data_dir + 'driving_log2.csv' # Track 2 First Run (Two Lap)
data_file3 = data_dir + 'driving_log3.csv' # Track 2 Correcting lap
data_file4 = data_dir + 'driving_log4.csv' # Track 2 Correcting lap
data_file5 = data_dir + 'driving_log5.csv' # Track 2 Correcting lap

def get_data():
    samples = []
    
    with open(data_file1) as csvfile: #currently after extracting the file is present in this path
        reader = csv.reader(csvfile)
        next(reader, None) #this is necessary to skip the first record as it contains the headings
        for line in reader:
            samples.append(line)
            
    with open(data_file3) as csvfile: #currently after extracting the file is present in this path
        reader = csv.reader(csvfile)
        next(reader, None) #this is necessary to skip the first record as it contains the headings
        for line in reader:
            samples.append(line)
   
    with open(data_file5) as csvfile: #currently after extracting the file is present in this path
        reader = csv.reader(csvfile)
        next(reader, None) #this is necessary to skip the first record as it contains the headings
        for line in reader:
            samples.append(line)
            
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

def generator(samples, batch_size=32):
    
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = data_dir+'./IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])
                    
                    # Append image that could center, left or right
                    images.append(image)
                    
                    # Compute stering angle based on the center angle
                    if (i == 0):
                        angles.append(center_angle)
                    elif (i == 1):
                        angles.append(center_angle + 0.3)
                    elif (i == 2):
                        angles.append(center_angle - 0.3)
                                            
                    # Augment data set
                    if (i == 0):
                        images.append(cv2.flip(image,1))
                        angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def create_model(loss='mse', optimizer='adam'):
    model = Sequential()
    
    # Normalization
    model.add(Lambda(lambda x:  (x / 255.0) - 0.5, input_shape=(160,320,3)))

    # Image Cropping
    model.add(Cropping2D(cropping=((70,25),(0,0))))           

    # Layers 1 - 5: Convolutional Layers
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))

    # Layers 6 - 8: Fully Connected Layers with Dropout
    model.add(Flatten())        
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)

    return model

def main():
    # Training Pipeline

    # 1. Get the training and validation data
    batch_size = 64
    train_samples, validation_samples = get_data()
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    print("train_generator: ", len(train_samples))
    print("validation_generator: ", len(validation_samples))
    
    # Create Model
    model = create_model()
    model.summary()
    
    # Train Model
    model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

    # Save Trained Model
    model.save('model.h5')
    
if __name__ == '__main__':
    main()
