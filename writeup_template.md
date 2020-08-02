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

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5]
![alt text][image6] ![alt text][image7]
![alt text][image8]

#### 2. Model's predictions 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No Entry     			| Priority road									|
| Yield					| Yield											|
| 50 km/h	      		| Yield							 				|
| Wild Animals Crossing | Bicycles crossing								|


#### 3. Softmax Probabilities

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| No Entry 										|
| .05					| Yield											|
| .04	      			| 50 km/h						 				|
| .01				    | Wild Animals Crossing							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


