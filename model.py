import csv
import cv2
import numpy as np
import math
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


lines = []
with open('driving_log2.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Data visualization
A = [x[3] for x in lines]
X = []
y = []

for j in range(-10,10):
    for i in A:
        if (j/10 + 0.1)  >= float(i) >= j/10:
            p = (j / 10 + 0.05)*100
            p = int(p)
            X.append(p)
print(X)
X = np.array(X)
print(X)

for k in [-95, -85, -75, -64, -54, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95]:
    y.append(np.size(np.where(X == k)))

A = [-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

fig = plt.figure()
plt.bar(A,y,0.07,color="Blue")
plt.xlabel("Angles")
plt.ylabel("Numbers")
plt.title("Distribution In Training Set")
plt.show()

# generator
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                steering = float(batch_sample[3])
                which_camera = np.random.choice([0,1,2])
                filp_prop = 0.5
                if which_camera == 1:
                    steering += 0.25
                elif which_camera == 2:
                    steering -= 0.25
                if filp_prop > 0.5:
                    steering = -1 * steering
                    center_image = cv2.flip(center_image, 1)
                images.append(center_image)
                angles.append(steering)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# model structure
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#煞笔李雨竹
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
#NVIDIA Structure
#I did a little change in original NVIDIA structure.
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))  ##Conv1: Input = 3*90*320, output = 24*43*158
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))  ##Conv2: Output = 36*20*77
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))  ##Conv3: Output = 48*8*37
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, subsample=(2,2), activation='relu'))  ##Conv4: Output = 64*3*18
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation='relu'))                   ##Conv5: Output = 64*1*16
model.add(Dropout(0.5))
model.add(Flatten())                                                   ##Output = 1046
model.add(Dense(100))                                                  ##Output = 100
model.add(Dense(50))                                                   ##Output = 50
model.add(Dense(10))                                                   ##Output = 10
model.add(Dense(1))                                                    ##Output = 1

model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
print('model saved')

# Loss
print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean square error')
plt.ylabel('mean square error loss')
plt.xlabel("epoch")
plt.legend(['training set', 'validation set'], loc = 'upper right')
plt.show()