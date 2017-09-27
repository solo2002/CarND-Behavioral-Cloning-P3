import cv2
import matplotlib.pyplot as plt
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
# def resize_image(dim, image):
#   img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#   return img
# #dim = (80, 160)
# image1 = cv2.imread('test.jpg')
# print(image1.shape)
# # cv2.imshow("resized",image1)
# # cv2.waitKey(0)
# #plt.imshow(image1)

# r = 480 / image1.shape[1]
# #dim = (270, int(image1.shape[0] * r))
# dim = (480,270)
# image2 = resize_image(dim, image1)
# cv2.imshow("resized", image2)
# print(image2.shape)
# cv2.waitKey(0)


# This function read csv file
def read_csv(path, lines):
  with open(path) as csvfile:
    reader = csv.reader(csvfile)
    # patch images' paths
    for line in reader:
      lines.append(line)
  # remove header
  #lines.pop(0)

# This function resize the image 
def resize_image(dim, image):
  img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  return img

# This function collect center image from selected folder
def collect_center_img(folder, lines, images, measurements):
  for line in lines:
  #for i in range(3):
    source_path = line[0] # lines[0][0]: center; lines[0][1]:left; lines[0][2]:right
    # content of lines[0]:
    #['IMG/center_2016_12_01_13_30_48_287.jpg', ' IMG/left_2016_12_01_13_30_48_287.jpg', 
    #' IMG/right_2016_12_01_13_30_48_287.jpg', ' 0', ' 0', ' 0', ' 22.14829']

    # file_name = source_path.split('/')[-1]
    # current_path = folder + file_name
    # image = cv2.imread(current_path)
    # dim = (160, 80)
    # #image2 = resize_image(dim, image)
    # images.append(image)
    # measurement = float(line[3]) # steering measurement
    # measurements.append(measurement)

     # select left
    left_name = line[1].split('/')[-1]
    left_image = cv2.imread(left_name)
    left_angle = float(line[3]) + 0.36
    images.append(left_image)
    measurements.append(left_angle)
    # images.append(cv2.flip(left_image,1))
    # angles.append(left_angle* (-1.0))

  print('Shape of image ', images[-1].shape)

# Read images in data folder
lines = []
read_csv('all_data/all.csv', lines)
print("length of lines ",len(lines))

images = []
measurements = []
collect_center_img('all_data/IMG/', lines, images, measurements)
print("length of images ",len(images))

# Read images in reverse folder
# lines = []
# read_csv('reverse/driving_log.csv', lines)
# print("length of lines ",len(lines))

# collect_center_img('reverse/IMG/', lines, images, measurements)
# print("length of images ",len(images))
