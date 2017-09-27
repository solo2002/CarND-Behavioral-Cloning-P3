import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda,Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle

# This function read csv file
def read_csv(path, lines):
  with open(path) as csvfile:
    reader = csv.reader(csvfile)
    # patch images' paths
    for line in reader:
      lines.append(line)
  # remove header
  #lines.pop(0)

def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5 + np.random.uniform(0.5, 1)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

samples = [] # lines

read_csv('data/driving_log.csv', samples)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("train_samples", len(train_samples))
print("validation_samples",len(validation_samples))
def generator(samples, batch_size=32):
    num_samples = len(samples)
    samples = sklearn.utils.shuffle(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]  
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # # Augmenting data
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle * (-1.0))

                # select left
                left_name = 'data/IMG/'+batch_sample[1].split('/')[-1]  
                left_image = cv2.imread(left_name)
                left_angle = float(batch_sample[3]) + 0.25
                images.append(left_image)
                angles.append(left_angle)
                images.append(cv2.flip(left_image,1))
                angles.append(left_angle* (-1.0))
                
                # select right
                right_name = 'data/IMG/'+batch_sample[2].split('/')[-1]  
                right_image = cv2.imread(right_name)
                right_angle = float(batch_sample[3]) - 0.25
                images.append(right_image)
                angles.append(right_angle)
                images.append(cv2.flip(right_image,1))
                angles.append(right_angle* (-1.0))

                prob = np.random.random()
                if prob > 0.5:
                  images.append(augment_brightness(center_image))
                  angles.append(center_angle)
                  images.append(augment_brightness(left_image))
                  angles.append(left_angle)
                  images.append(augment_brightness(right_image))
                  angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
# Add a Lambda layer for normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Cropping 
model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(... finish defining the rest of your model architecture here ...)

# # # LeNet
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

# Nvidia Architecture
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=
  len(train_samples), validation_data=validation_generator,
  nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')


print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
