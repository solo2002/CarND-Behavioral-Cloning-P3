# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing: 
```
python drive.py model.h5
```
The video of the project is uploaded to the youtube: https://youtu.be/tbxJqSKXMdw

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Nvidia pipeline is used in this project with minor modifications. Final model consisted of the following layers:
 
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image (normalized to [-0.5, 0.5], a Keras lambda layer)   							| 
| Convolution 5x5     	| stride: 2x2, depth: 24, subsample: 2x2, activation: Relu|
| Convolution 5x5     	| stride: 2x2, depth: 36, subsample: 2x2, activation: Relu|
| Convolution 5x5     	| stride: 2x2, depth: 48, subsample: 2x2, activation: Relu|
| Convolution 3x3     	| stride: non, depth: 64, activation: Relu|
| Convolution 3x3     	| stride: non, depth: 64, activation: Relu|										|
| Flatten		| Output = 400        									|
| Fully connected		| Output = 120        									|
| Fully connected		| Output = 84       									|
| Fully connected		| Output = 1       									|
 
* Model Training

  (1) Batch size: since I am using my personal laptop, I decrease the batch size to 32.
  
  (2) Learning rate: 0.001, 0.005, and 0,01 are tested, and 0.001 is best.
  
  (3) Epochs: 10, 20, 30, and 40 are examined, and 30 is selected.
  
  (4) The range of cropped: 3, 4, and 5 pixels from each edge are tested, and cropping 4 pixels shows the best performance.
  
  (5) Brightness augmenting: different levels of brightness augmenting are tested, but none of them shows significant improvement.
  
* Solution Approach
  This trained model via using parameter-setting mentioned above gets 0.940 accuracy on validation data and 0.922 accuracy on testing data.
* Use the model to make predictions on 6 new images, please refer to the ipynb or its html file those images.
  The prediction accuracy is 1.00.
  Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h)   									| 
| Priority road     			| Priority road 										|
| No entry        		| No entry    									| 
| Turn right ahead					| Turn right ahead											|
| Go straight or right     		| Go straight or right				 				|
| Roundabout mandatory			| Roundabout mandatory      							|

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

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
