# **Behavioral Cloning**

## Writeup for Project 3 Final

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./submit_Images/Architecture.PNG "Model Visualization"
[image2]: ./submit_Images/center.jpg "Center image"
[image3]: ./submit_Images/left_recovery.jpg "Recovery Image"
[image4]: ./submit_Images/right_recovery.jpg "Recovery Image"
[image5]: ./submit_Images/before_flip.png "Recovery Image"
[image6]: ./submit_Images/after_flip.png "Normal Image"
[image7]: ./submit_Images/BGR_image.png "Flipped Image"
[image8]: ./submit_Images/RGB_image.png "Flipped Image"
[image9]: ./submit_Images/train_val_curve.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_P3_FengChen.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with (3x3) and (5x5) filter sizes and depths between 24 and 64 (model.py lines 86-87)

The model includes ELU layers to introduce nonlinearity (code line 109), and the data is normalized in the model using a Keras lambda layer (code line 94).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 113).
A problem found with dropout is that, if too much added, e.g., dropout behind each convolution layer, the fitting result is very bad, and car drives on the edge line.

The model was trained and validated on different data sets to ensure that the model was not overfitting.
Here instead of using two different files, I already combined my own generated training data with Udacity sample
training data. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, but I was able to manually set the learning rate (model.py line 119). Later on I decide it is not necessary so I used the adam model.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving,
recovering from the left and right sides of the road. For the steering angle, I added 0.25 for the left side and
-0.25 for the right side.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the well qualified model,
instead of building everything from scratch.

My first step was to use a convolution neural network model similar to the LeNet.
I thought this model might be appropriate because it is also about feature recognition.

In order to gauge how well the model was working, I split my image and steering angle
data into a training and validation set. I found that my first model had a low mean
squared error on the training set but a high mean squared error on the validation set.
This implied that the model was overfitting.

To combat the overfitting, I modified the model so that the validation error can be smaller than fitting loss.

Then I added the dropout layer after each convolution, I found that my model is underfitting. The validation error became smaller than the fitting loss.

I removed all the dropout layers behind convolutions, and only kept the dropout behind the Flatten layer.

The final step was to run the simulator to see how well the car was driving around track one.
There were a few spots where the vehicle fell off the track, especially when the road has big turns.
To improve the driving behavior in these cases, I first tried collecting more data. It helps a little bit but not much.
Another thing I tried is tune the running speed in drive.py. I found the control wasn't idea when driving around the big corners.
With increasing the driving speed from 9 to 15, my car is okay.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 91-117) consisted of a convolution
neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.
 Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that
the vehicle would learn to back to track. These images show what a recovery looks like starting from right side of
the road and left side of the road:

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

Another important thing is change the color when import the image. Since this is a colored image (not changed to gray scale)
drive.py read as RGB image, but cv2 read as BGR image.

Here is the difference:
![alt text][image7]
![alt text][image8]

After the collection process, I had 9000 number of data points. I then preprocessed this data by generator to save memory


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model.
The validation set helped determine if the model was over or under fitting.
The ideal number of epochs was 5 as evidenced by the loss curve show as below. Here you can see that the validation error is trying to go down below training error.
![alt text][image9]
I used an adam optimizer so that manually training the learning rate wasn't necessary.
