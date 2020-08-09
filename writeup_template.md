# **Traffic Sign Recognition** 

## Writeup

### Classification of German Traffic Signs with usage of LeNet Neural Network

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Data_Images/Image1.png "Visualization Data Random Images"
[image2]: ./Data_Images/Image2.png "Visualization Data Bar Chart"
[image3]: ./Data_Images/Image3.png "German Traffic Signs from Web"
[image4]: ./Web_Test_Images/No_Entry.jfif "No_Entry"
[image5]: ./Web_Test_Images/SpeedLimit_50.jfif "SpeedLimit_50"
[image6]: ./Web_Test_Images/Stop_Sign.jfif "Stop_Sign"
[image7]: ./Web_Test_Images/WildAnimalCrossing_Sign.jfif "WildAnimalCrossing_Sign"
[image8]: ./Web_Test_Images/Yield_Sign.jfif "Yield_Sign"
[image9]: ./Data_Images/Image4.png "Softmax Probabilities"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Data Set Summary & Exploration

#### 1. Dataset Summary

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

Following are 10 random images picked from "Training" data set:

![alt text][image1]

Here is an exploratory visualization of the data set. It is a bar chart showing number of images for each sign:

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Preprocess

I converted the images to grayscale
Then I normalized the image data


#### 2. Model Architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16 					|
| Flatten	        	| outputs 400 									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|
| RELU					|												|
 


#### 3. Way of Model Trained
To train the model, I used an Adam optimizer and following parameters:

* EPOCHS = 10
* BATCH_SIZE = 120
* Rate = 0.001
* mu = 0
* sigma = 0.1


#### 4. My final model results were:

* training set accuracy of 0.987
* validation set accuracy of 0.958
* test set accuracy of 0.859


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5]
![alt text][image6] ![alt text][image7]
![alt text][image8]

My input images was bright and clearly visible. I have resized the input images, converted 3 channels and applied Gaussian Blur to feed into LeNet


#### 2. Model's predictions 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No Entry     			| Priority road									|
| Yield					| Yield											|
| 50 km/h	      		| Yield							 				|
| Wild Animals Crossing | Bicycles crossing								|


The model was able to correctly guess 2 of 5 traffic signs, which has an accuracy of 40%. But the Test data accuracy of 85.9% shows that my model is overfitting and predicting other 3 images wrong.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For every image the softmax is predicting 100% correct even though only 2 (the first "Stop Sign" and third "Yield") out of 5 images are correct. Probably I need add dropout or do max pooling.

![alt text][image9]